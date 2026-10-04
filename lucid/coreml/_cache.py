"""Compile a package once, and open that compiled model every time after.

Core ML runs a compiled model (``.mlmodelc``), and the first time one is
loaded it is specialised into a bundle under the user's caches —
``~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache`` — which for a GPU
segment holds a full copy of the weights: 1.1 GB for YOLO v1.  That bundle
is keyed by the compiled model's path *and* its file identity.  Measured on
macOS 27: opening the same ``.mlmodelc`` again, from the same process or
another, adds nothing; a copy of it adds a new bundle, even one put back at
the very same path.

Compiling on every load therefore wrote a bundle on every load that nothing
would read again.  A development machine's cache reached 121 GB, and the
test suite pushed the disk past macOS's near-low mark, where the purge that
followed aborted a process from inside Core ML (CHA-22, see
``lucid/test/unit/coreml/conftest.py``).

So the compiled model is kept — one per package *content* — and every load
of that content opens the same directory.

Key
    SHA-256 over every file in the package (relative path, size, bytes),
    after the Core ML build and the OS build — a system update compiles
    afresh instead of reusing an older compiler's output.  Content, not
    path: a package rewritten in place with new weights is a new entry, and
    the same package copied elsewhere, or exported again from the same
    model, is the same one.
Location
    ``LUCID_COREML_CACHE_DIR`` when set; else ``$LUCID_HOME/coreml``, beside
    the weights cache that variable already moves; else
    ``~/.cache/lucid/coreml``.
Bound
    ``LUCID_COREML_CACHE_LIMIT`` bytes (``K``/``M``/``G``/``T`` suffixes
    accepted), 4 GiB by default — three models the size of YOLO v1.
    Inserting past it removes the least recently used entries, except the
    one just made and any entry a live handle holds, in this process or
    another, so a single model larger than the limit is still compiled once
    and reused while it is open.  ``0`` turns the cache off: each handle
    compiles privately and removes its copy when it closes — which, as
    before, leaves Core ML one bundle per load.
    :func:`lucid.coreml.empty_cache` removes every entry no handle holds.
Concurrency
    A compile is moved into a staging directory inside the cache and
    renamed onto its entry.  ``rename`` is atomic and refuses a non-empty
    target, so of two processes that compile the same package at once one
    wins and the other discards its copy and opens the winner's; two
    threads of one process do not both compile.  Every handle holds a
    shared ``flock`` on its entry's lock file for as long as it lives;
    eviction asks for an exclusive one without waiting and passes over what
    it cannot have.  The kernel drops a dead process's locks, so a crash
    leaves nothing held.
Disk
    Core ML does not fail cleanly when the disk fills: it ends the process,
    ``LLVM ERROR: IO failure on output stream: No space left on device``.
    Before compiling or loading, each volume the step writes to is checked
    for what it will write — the compiled model where Core ML compiles it
    and where it is kept, a bundle the size of the package where Core ML
    keeps those — and a shortfall raises :class:`OSError` (``ENOSPC``).

Not bounded here: the bundle Core ML made from an entry that is later
evicted stays in Core ML's cache until macOS purges it.  Its directory names
cannot be traced back to the entry, and Core ML's cache is not Lucid's to
prune.  Reopening a package — the case that grew without bound — writes
none.
"""

import errno
import fcntl
import functools
import hashlib
import json
import os
import plistlib
import shutil
import tempfile
import threading
import time
import uuid
import warnings

from lucid._C import engine as _C_engine

__all__ = ["Lease", "empty_cache", "open_compiled"]

#: Bumped when the layout of an entry changes, so an older layout is never
#: mistaken for a current one; it ages out like any unused entry.
_FORMAT = 1

#: The bound when ``LUCID_COREML_CACHE_LIMIT`` is unset.
_DEFAULT_LIMIT = 4 << 30

#: Kept free on every volume beyond what a step is estimated to write.  The
#: estimate is the package's size; what Core ML writes for it is near that
#: rather than exactly it.
_HEADROOM = 256 << 20

#: A staging or trash directory, or a lock with no entry, older than this
#: belongs to a process that died part way; a younger one may be in use.
_ABANDONED_AFTER = 3600.0

_SUFFIX = ".mlmodelc"

#: Guards the dictionaries below.
_lock = threading.Lock()
#: One eviction at a time in this process.
_evicting = threading.Lock()
#: One lock per key, so two threads loading the same new package compile
#: it once rather than racing to the rename.
_compiling: dict[str, threading.Lock] = {}
#: ``realpath -> (stat signature, key, bytes)``, so a package opened again
#: unchanged in the same process is not read again.
_known: dict[str, tuple[tuple[tuple[str, int, int, int], ...], str, int]] = {}
#: Roots this process already warned it could not use.
_unusable: set[str] = set()


class Lease:
    """What a handle opens, held for as long as the handle needs it.

    ``path`` is the cached ``.mlmodelc`` to open, and releasing the lease
    drops the shared lock that keeps eviction away from it.  With the
    cache off it is the package itself: the engine compiles that privately
    and removes its copy when the last use of the handle ends — after a
    prediction still running on another thread, which a removal here could
    not wait for.  Safe to release twice.
    """

    __slots__ = ("_fd", "path")

    def __init__(self, path: str, fd: int | None) -> None:
        self.path = path
        self._fd = fd

    @property
    def held(self) -> bool:
        """Whether this lease still keeps a cached compiled model in place."""
        return self._fd is not None

    def release(self) -> None:
        """Let the compiled model go."""
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)


def cache_dir() -> str:
    """Where compiled models are kept, whether or not it exists yet.

    Returns
    -------
    str
        ``LUCID_COREML_CACHE_DIR`` when set; else ``$LUCID_HOME/coreml``,
        beside the weights cache that variable already moves; else
        ``~/.cache/lucid/coreml``. Read on every call, so a change to the
        environment takes effect at the next load.
    """
    chosen = os.environ.get("LUCID_COREML_CACHE_DIR")
    if chosen:
        return os.path.abspath(os.path.expanduser(chosen))
    home = os.environ.get("LUCID_HOME")
    if home is not None:
        return os.path.abspath(os.path.join(os.path.expanduser(home), "coreml"))
    return os.path.join(os.path.expanduser("~"), ".cache", "lucid", "coreml")


def cache_limit() -> int:
    """The bound the cache keeps to, in bytes.

    Returns
    -------
    int
        ``LUCID_COREML_CACHE_LIMIT`` read as a size — bytes, or a number
        with ``K``/``M``/``G``/``T`` (binary; ``B`` or ``iB`` optional) —
        or 4 GiB when it is unset. ``0`` means the cache is off.

    Raises
    ------
    ValueError
        The variable is set to something that is not a size.
    """
    raw = os.environ.get("LUCID_COREML_CACHE_LIMIT", "").strip()
    if not raw:
        return _DEFAULT_LIMIT
    text = raw.upper().removesuffix("B").removesuffix("I")
    scale = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}.get(text[-1:], 1)
    try:
        value = float(text[:-1] if scale > 1 else text)
    except ValueError:
        value = -1.0
    if not 0 <= value < float("inf"):
        raise ValueError(
            f"lucid.coreml: LUCID_COREML_CACHE_LIMIT={raw!r} is not a size — give "
            f"bytes, optionally with K, M, G or T (4G), or 0 to turn the cache off"
        )
    return int(value * scale)


def _core_ml_home() -> str:
    """The home Core ML keeps its bundles under for this process."""
    return os.environ.get("CFFIXED_USER_HOME") or os.path.expanduser("~")


@functools.cache
def _fingerprint() -> str:
    """What produced a compiled model, beyond the package it came from."""

    def read(path: str, key: str) -> str:
        try:
            with open(path, "rb") as handle:
                return str(plistlib.load(handle).get(key, "unknown"))
        except OSError, ValueError:
            return "unknown"

    coreml = read(
        "/System/Library/Frameworks/CoreML.framework/Resources/version.plist",
        "CFBundleVersion",
    )
    system = read(
        "/System/Library/CoreServices/SystemVersion.plist", "ProductBuildVersion"
    )
    return f"lucid-coreml-cache/{_FORMAT};coreml={coreml};os={system}"


def _files(package: str) -> list[tuple[str, str]]:
    """Every file in the package as ``(relative path, absolute path)``."""
    if os.path.isfile(package):
        return [(os.path.basename(package), package)]
    found: list[tuple[str, str]] = []
    for here, dirs, names in os.walk(package):
        dirs.sort()
        for name in sorted(names):
            full = os.path.join(here, name)
            found.append((os.path.relpath(full, package), full))
    return found


def _identify(package: str) -> tuple[str, int]:
    """The package's cache key and its size in bytes."""
    files = _files(package)
    stamp = []
    for relative, full in files:
        info = os.stat(full)
        stamp.append((relative, info.st_size, info.st_mtime_ns, info.st_ino))
    signature = tuple(stamp)
    real = os.path.realpath(package)
    with _lock:
        known = _known.get(real)
    if known is not None and known[0] == signature:
        return known[1], known[2]

    digest = hashlib.sha256(_fingerprint().encode())
    size = 0
    chunk = bytearray(8 << 20)
    view = memoryview(chunk)
    for (relative, full), (_r, length, _m, _i) in zip(files, signature):
        size += length
        digest.update(f"\0{relative}\0{length}\0".encode())
        with open(full, "rb") as handle:
            while read := handle.readinto(chunk):
                digest.update(view[:read])
    key = digest.hexdigest()
    with _lock:
        _known[real] = (signature, key, size)
    return key, size


def _free_bytes(path: str) -> int:
    """Free space on the volume holding ``path``; replaced in tests."""
    return shutil.disk_usage(path).free


def _existing(path: str) -> str:
    """``path``, or the nearest ancestor of it that exists."""
    path = os.path.abspath(path)
    while not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    return path


def _human(count: int) -> str:
    if count < 1 << 20:
        return f"{count / (1 << 10):.1f} KB"
    if count < 1 << 30:
        return f"{count / (1 << 20):.1f} MB"
    return f"{count / (1 << 30):.2f} GB"


def _require_room(package: str, size: int, *, compile_into: str | None) -> None:
    """Raise ``ENOSPC`` unless every volume the load writes to has room.

    ``compile_into`` is where the compiled model will be kept, or ``None``
    when it is already there and only Core ML's bundle may still be written.
    """
    writes: list[tuple[str, str]] = []
    if compile_into is not None:
        scratch = tempfile.gettempdir()
        writes.append((scratch, "the compiled model"))
        if (
            os.stat(_existing(compile_into)).st_dev
            != os.stat(_existing(scratch)).st_dev
        ):
            writes.append((compile_into, "the compiled model's kept copy"))
    writes.append(
        (
            os.path.join(_core_ml_home(), "Library", "Caches"),
            "the bundle Core ML specialises it into",
        )
    )

    volumes: dict[int, tuple[str, int, list[str]]] = {}
    for where, what in writes:
        anchor = _existing(where)
        device = os.stat(anchor).st_dev
        first, total, whats = volumes.get(device, (anchor, 0, []))
        volumes[device] = (first, total + size, [*whats, what])

    short = []
    for anchor, needed, whats in volumes.values():
        free = _free_bytes(anchor)
        if free < needed + _HEADROOM:
            short.append(
                f"{_human(needed + _HEADROOM)} free on the volume holding {anchor} "
                f"({' and '.join(whats)}, plus {_human(_HEADROOM)} to spare) and "
                f"{_human(free)} is"
            )
    if short:
        raise OSError(
            errno.ENOSPC,
            f"lucid.coreml: loading {package} ({_human(size)}) needs "
            + "; it also needs ".join(short)
            + ". Core ML does not fail cleanly when the disk fills — it ends "
            "the process — so the load stops here instead. Free some space, "
            "or point LUCID_COREML_CACHE_DIR at a volume that has it.",
        )


def _hold(lock_path: str, *, exclusive: bool) -> int | None:
    """An fd holding ``lock_path`` shared, or exclusive if free right now.

    ``None`` when an exclusive lock was asked for and someone holds it.  An
    eviction can unlink the lock file while this waits on it, and a lock on
    an unlinked file guards nothing, so the inode is checked and the lock
    taken again on the file that is there now.
    """
    operation = (fcntl.LOCK_EX | fcntl.LOCK_NB) if exclusive else fcntl.LOCK_SH
    while True:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, operation)
        except BlockingIOError:
            os.close(fd)
            return None
        except BaseException:
            os.close(fd)
            raise
        try:
            current = os.stat(lock_path).st_ino
        except FileNotFoundError:
            current = -1
        if current == os.fstat(fd).st_ino:
            return fd
        os.close(fd)


def _is_whole(entry: str) -> bool:
    return os.path.isfile(os.path.join(entry, "coremldata.bin"))


def _tree_bytes(path: str) -> int:
    total = 0
    for here, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.lstat(os.path.join(here, name)).st_size
            except OSError:
                pass
    return total


def _unlink(*paths: str) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _write_note(root: str, key: str, package: str) -> None:
    """Record the entry's size and source; its mtime is when it was last used."""
    size = _tree_bytes(os.path.join(root, f"{key}{_SUFFIX}"))
    scratch = os.path.join(root, f".note-{uuid.uuid4().hex}")
    with open(scratch, "w") as handle:
        json.dump({"bytes": size, "package": package, "created": time.time()}, handle)
    os.replace(scratch, os.path.join(root, f"{key}.json"))


def _sweep(root: str, name: str, now: float) -> None:
    """Remove what a process that died part way left behind."""
    full = os.path.join(root, name)
    try:
        if now - os.lstat(full).st_mtime <= _ABANDONED_AFTER:
            return
    except OSError:
        return
    if not name.endswith(".lock"):
        if os.path.isdir(full):
            shutil.rmtree(full, ignore_errors=True)
        else:
            _unlink(full)
        return
    # A lock whose compile never produced an entry.
    fd = _hold(full, exclusive=True)
    if fd is None:
        return
    try:
        key = name.removesuffix(".lock")
        if not os.path.exists(os.path.join(root, f"{key}{_SUFFIX}")):
            _unlink(full, os.path.join(root, f"{key}.json"))
    finally:
        os.close(fd)


def _entries(root: str) -> list[tuple[float, int, str]]:
    """Every entry as ``(last used, bytes, key)``."""
    found: list[tuple[float, int, str]] = []
    try:
        names = os.listdir(root)
    except FileNotFoundError:
        return found
    present = set(names)
    now = time.time()
    for name in names:
        if name.startswith((".staging-", ".trash-", ".note-")):
            _sweep(root, name, now)
            continue
        if name.endswith(".lock"):
            if f"{name.removesuffix('.lock')}{_SUFFIX}" not in present:
                _sweep(root, name, now)
            continue
        if not name.endswith(_SUFFIX) or name.startswith("."):
            continue
        key = name.removesuffix(_SUFFIX)
        note = os.path.join(root, f"{key}.json")
        try:
            used = os.stat(note).st_mtime
            with open(note) as handle:
                size = int(json.load(handle)["bytes"])
        except OSError, ValueError, KeyError, TypeError:
            try:
                used = os.stat(os.path.join(root, name)).st_mtime
            except OSError:
                continue
            size = _tree_bytes(os.path.join(root, name))
        found.append((used, size, key))
    return found


def _remove(root: str, key: str) -> bool:
    """Remove one entry unless a handle holds it; whether it went."""
    lock_path = os.path.join(root, f"{key}.lock")
    fd = _hold(lock_path, exclusive=True)
    if fd is None:
        return False
    trash = os.path.join(root, f".trash-{uuid.uuid4().hex}")
    try:
        try:
            os.rename(os.path.join(root, f"{key}{_SUFFIX}"), trash)
        except FileNotFoundError:
            trash = ""
        _unlink(os.path.join(root, f"{key}.json"), lock_path)
    finally:
        os.close(fd)
    if trash:
        shutil.rmtree(trash, ignore_errors=True)
    return True


def _evict(root: str, limit: int, keep: str | None) -> int:
    """Remove least recently used entries until the cache fits ``limit``."""
    with _evicting:
        entries = sorted(_entries(root))
        total = sum(size for _used, size, _key in entries)
        freed = 0
        for _used, size, key in entries:
            if total <= limit:
                break
            if key == keep or not _remove(root, key):
                continue
            total -= size
            freed += size
        return freed


def _private(package: str, size: int) -> Lease:
    """The package itself, for the engine to compile for one handle only."""
    if size:
        _require_room(package, size, compile_into=tempfile.gettempdir())
    return Lease(package, None)


def _usable(root: str) -> bool:
    try:
        os.makedirs(root, exist_ok=True)
        if not os.access(root, os.W_OK | os.X_OK):
            raise PermissionError(errno.EACCES, "not writable")
        return True
    except OSError as exc:
        with _lock:
            first = root not in _unusable
            _unusable.add(root)
        if first:
            warnings.warn(
                f"lucid.coreml: cannot keep compiled models in {root} ({exc}); "
                f"compiling on every load instead, which leaves Core ML a new "
                f"cache bundle each time. Set LUCID_COREML_CACHE_DIR to a "
                f"writable directory.",
                RuntimeWarning,
                stacklevel=5,
            )
        return False


def _insert(root: str, key: str, package: str, size: int) -> None:
    """Compile ``package`` onto its entry, unless another process just did."""
    _require_room(package, size, compile_into=root)
    entry = os.path.join(root, f"{key}{_SUFFIX}")
    compiled = _C_engine.coreml.compile_model(package)
    staging = os.path.join(root, f".staging-{uuid.uuid4().hex}")
    try:
        shutil.move(compiled, staging)
        if os.path.isdir(entry) and not _is_whole(entry):
            # Never left by this module — a rename is whole or absent — so
            # something else damaged it; clear it rather than open it.
            shutil.rmtree(entry, ignore_errors=True)
        try:
            os.rename(staging, entry)
        except OSError as exc:
            # Another process's copy got there first; open that one.
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(compiled, ignore_errors=True)
    _write_note(root, key, package)


def open_compiled(package: str) -> Lease:
    """The compiled model for ``package``, compiling it only if no one has.

    Parameters
    ----------
    package : str
        The ``.mlpackage`` to open.

    Returns
    -------
    Lease
        Holds the cached ``.mlmodelc`` until released — or, with the cache
        off or unusable, names the package for the engine to compile.

    Raises
    ------
    OSError
        ``ENOSPC`` when a volume the load writes to lacks the room.
    RuntimeError
        Core ML could not compile the package.
    """
    if not os.path.exists(package):
        # Core ML's own error names the missing path; nothing to cache.
        return _private(package, 0)
    limit = cache_limit()
    root = cache_dir()
    if limit == 0 or not _usable(root):
        size = sum(os.stat(full).st_size for _relative, full in _files(package))
        return _private(package, size)

    key, size = _identify(package)
    entry = os.path.join(root, f"{key}{_SUFFIX}")
    fd = _hold(os.path.join(root, f"{key}.lock"), exclusive=False)
    if fd is None:  # a shared lock waits rather than refusing
        raise RuntimeError("lucid.coreml: could not lock the compile cache")
    inserted = False
    try:
        with _lock:
            serial = _compiling.setdefault(key, threading.Lock())
        with serial:
            if _is_whole(entry):
                _require_room(package, size, compile_into=None)
                try:
                    os.utime(os.path.join(root, f"{key}.json"))
                except FileNotFoundError:
                    _write_note(root, key, package)
            else:
                _insert(root, key, package, size)
                inserted = True
    except BaseException:
        os.close(fd)
        raise
    lease = Lease(entry, fd)
    if inserted:
        _evict(root, limit, keep=key)
    return lease


def empty_cache() -> int:
    """Remove every cached compiled model that no handle holds.

    An entry a live handle holds — in this process or another — keeps its
    shared lock, and is passed over.

    Returns
    -------
    int
        Bytes freed; ``0`` when the cache is empty or does not exist.
    """
    root = cache_dir()
    if not os.path.isdir(root):
        return 0
    return _evict(root, 0, keep=None)
