import argparse
import contextlib
import io
import time
import statistics
import math

import lucid
import matplotlib.pyplot as plt


@contextlib.contextmanager
def silence_stdout(enabled: bool):
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def build_graph(n: int, depth: int, device: str, seed: int):
    lucid.random.seed(seed)

    # overflow 방지
    x = lucid.random.randn(n, n) * 0.01
    x = x.to(device)
    x.requires_grad = True

    y = x
    for _ in range(depth):
        y = lucid.log(lucid.exp(y))  # LogExp fuse target

    loss = y.sum()
    return x, loss


def measure_one(
    n: int,
    depth: int,
    device: str,
    iters: int,
    warmup: int,
    enable_fusion: bool,
    silence: bool,
):
    lucid.ENABLE_FUSION = enable_fusion

    # warmup
    with silence_stdout(silence):
        for i in range(warmup):
            x, loss = build_graph(n, depth, device, seed=1000 + i)
            loss.backward(retain_graph=False)
            if device == "gpu":
                x.eval()

    times = []
    with silence_stdout(silence):
        for i in range(iters):
            x, loss = build_graph(n, depth, device, seed=2000 + i)

            t0 = time.perf_counter()
            loss.backward(retain_graph=False)
            if device == "gpu":
                x.eval()  # lazy sync
            t1 = time.perf_counter()

            times.append(t1 - t0)

    return statistics.mean(times)


def make_sizes_10_points(n_min: int, n_max: int):
    """
    10 points between n_min and n_max (inclusive-ish).
    log-spaced then rounded to int and uniqued.
    """
    if n_min < 1:
        n_min = 1
    if n_max < n_min:
        n_max = n_min

    pts = []
    for k in range(10):
        r = k / 9.0
        n = int(round(n_min * ((n_max / n_min) ** r)))
        pts.append(max(1, n))

    # uniq + sort (keep about 10)
    pts = sorted(set(pts))
    # if rounding collapsed too much, fill linearly
    while len(pts) < 10:
        step = max(1, (n_max - n_min) // 9)
        pts = sorted(set([n_min + i * step for i in range(10)]))
    return pts[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmin", type=int, default=32, help="min matrix size n (n x n)")
    ap.add_argument("--nmax", type=int, default=1024, help="max matrix size n (n x n)")
    ap.add_argument("--depth", type=int, default=50, help="number of (exp->log) pairs")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "--silence", action="store_true", help="silence stdout during timing"
    )
    ap.add_argument(
        "--save", type=str, default="", help="optional path to save plot (png)"
    )
    ap.add_argument(
        "--epochs", type=int, default=3, help="repeat full sweep this many times"
    )
    args = ap.parse_args()
    if args.device == "gpu":
        # Trigger Metal availability checks early (will raise if unavailable)
        _ = lucid.Tensor(0.0, device="gpu")

    ns = make_sizes_10_points(args.nmin, args.nmax)
    xs = [n * n for n in ns]  # number of elements

    # Per-epoch curves
    t_off_epochs: list[list[float]] = []
    t_on_epochs: list[list[float]] = []

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        t_off: list[float] = []
        t_on: list[float] = []

        for idx, n in enumerate(ns):
            print(f"[{idx+1}/{len(ns)}] n={n} (elems={n*n:,}) ...")

            off = measure_one(
                n=n,
                depth=args.depth,
                device=args.device,
                iters=args.iters,
                warmup=args.warmup,
                enable_fusion=False,
                silence=args.silence,
            )
            on = measure_one(
                n=n,
                depth=args.depth,
                device=args.device,
                iters=args.iters,
                warmup=args.warmup,
                enable_fusion=True,
                silence=args.silence,
            )

            t_off.append(off)
            t_on.append(on)

            print(
                f"  OFF mean: {off:.6f}s | ON mean: {on:.6f}s | speedup: {off/on:.3f}x"
            )

        t_off_epochs.append(t_off)
        t_on_epochs.append(t_on)

    speedup_epochs = [
        [off / on for off, on in zip(t_off, t_on)]
        for t_off, t_on in zip(t_off_epochs, t_on_epochs)
    ]

    # Mean curves across epochs
    t_off_mean = [statistics.mean(vals) for vals in zip(*t_off_epochs)]
    t_on_mean = [statistics.mean(vals) for vals in zip(*t_on_epochs)]
    speedup_mean = [statistics.mean(vals) for vals in zip(*speedup_epochs)]

    # Variability bands (std across epochs)
    t_off_std = [
        statistics.pstdev(vals) if len(vals) > 1 else 0.0 for vals in zip(*t_off_epochs)
    ]
    t_on_std = [
        statistics.pstdev(vals) if len(vals) > 1 else 0.0 for vals in zip(*t_on_epochs)
    ]
    speedup_std = [
        statistics.pstdev(vals) if len(vals) > 1 else 0.0
        for vals in zip(*speedup_epochs)
    ]

    t_off_lo = [m - s for m, s in zip(t_off_mean, t_off_std)]
    t_off_hi = [m + s for m, s in zip(t_off_mean, t_off_std)]
    t_on_lo = [m - s for m, s in zip(t_on_mean, t_on_std)]
    t_on_hi = [m + s for m, s in zip(t_on_mean, t_on_std)]
    speedup_lo = [max(0.0, m - s) for m, s in zip(speedup_mean, speedup_std)]
    speedup_hi = [m + s for m, s in zip(speedup_mean, speedup_std)]

    # Find crossing point of mean curves (fusion OFF vs ON) in x-space (elements = n^2)
    cross_x: float | None = None
    cross_y: float | None = None
    diffs = [off - on for off, on in zip(t_off_mean, t_on_mean)]
    for i in range(len(xs) - 1):
        d0, d1 = diffs[i], diffs[i + 1]
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = t_off_mean[i], t_off_mean[i + 1]

        if d0 == 0.0:
            cross_x = float(x0)
            cross_y = float(y0)
            break

        # Sign change indicates a crossing between i and i+1
        if d0 * d1 < 0.0:
            # Linear interpolation in x for the zero of diff
            t = (0.0 - d0) / (d1 - d0)
            cross_x = float(x0 + t * (x1 - x0))
            cross_y = float(y0 + t * (y1 - y0))
            break

    cross_n: int | None = None
    if cross_x is not None:
        cross_n = max(1, int(round(math.sqrt(cross_x))))

    # Plot: left = absolute time, right = speedup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # Consistent colors
    c_off = "#1f77b4"  # matplotlib default blue
    c_on = "#ff7f0e"  # matplotlib default orange
    c_spd = "#2ca02c"  # matplotlib default green

    # Left: absolute backward time (all epochs, faint)
    for t_off, t_on in zip(t_off_epochs, t_on_epochs):
        ax1.plot(xs, t_off, marker="o", alpha=0.4, linewidth=1.0, color=c_off)
        ax1.plot(xs, t_on, marker="o", alpha=0.4, linewidth=1.0, color=c_on)

    # Left: shaded variability bands (±1 std across epochs)
    ax1.fill_between(xs, t_off_lo, t_off_hi, color=c_off, alpha=0.2, linewidth=0)
    ax1.fill_between(xs, t_on_lo, t_on_hi, color=c_on, alpha=0.2, linewidth=0)

    # Left: mean curves (bold)
    ax1.plot(
        xs,
        t_off_mean,
        marker="o",
        linewidth=2.5,
        color=c_off,
        label="fusion OFF (mean)",
    )
    ax1.plot(
        xs, t_on_mean, marker="o", linewidth=2.5, color=c_on, label="fusion ON (mean)"
    )

    # Mark crossing point of mean curves (if any)
    if cross_x is not None and cross_y is not None and cross_n is not None:
        ax1.axvline(cross_x, color="black", linewidth=1.2, alpha=0.85)
        ax1.plot([cross_x], [cross_y], marker="o", color="black")
        # Place label slightly above the dot (works with log y-scale)
        ax1.text(
            cross_x,
            cross_y * 1.15,
            f"n≈{cross_n}",
            color="black",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax1.set_xlabel(r"Number of elements ($n^2$)")
    ax1.set_ylabel("Mean backward time (seconds)")
    ax1.set_title(
        f"Backward time vs elements (depth={args.depth}, device={args.device})"
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Right: speedup (all epochs, faint)
    for spd in speedup_epochs:
        ax2.plot(xs, spd, marker="o", alpha=0.4, linewidth=1.0, color=c_spd)

    # Right: shaded variability band (±1 std across epochs)
    ax2.fill_between(xs, speedup_lo, speedup_hi, color=c_spd, alpha=0.2, linewidth=0)

    # Right: mean speedup (bold)
    ax2.plot(
        xs,
        speedup_mean,
        marker="o",
        linewidth=2.5,
        color=c_spd,
        label="speedup (mean, OFF/ON)",
    )

    ax2.set_xlabel(r"Number of elements ($n^2$)")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Fusion speedup")
    ax2.set_xscale("log")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
        print("\nSaved plot to:", args.save)

    plt.show()


if __name__ == "__main__":
    main()
