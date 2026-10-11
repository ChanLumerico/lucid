# Third-Party Notices

Lucid itself is distributed under the MIT License (see [LICENSE](LICENSE)).
This file lists the third-party works bundled in Lucid's source and binary
distributions, the files that contain them, and the license terms under which
they are redistributed.

## Cephes Mathematical Library

- Files: `lucid/_ops/composite/_bessel.py` (Bessel I0 / I1 / J0 / J1 / Y0 / Y1 /
  K0 / K1 coefficient tables)
- License: BSD-3-Clause

The notice below is reproduced verbatim from the "Name: Cephes" entry of the
SciPy xsf project's `LICENSES_bundled.txt`
(<https://raw.githubusercontent.com/scipy/xsf/main/LICENSES_bundled.txt>,
retrieved 2026-10-11).

```text
  Distributed under 3-clause BSD license with permission from the author,
  see https://lists.debian.org/debian-legal/2004/12/msg00295.html

  Cephes Math Library Release 2.8:  June, 2000
  Copyright 1984, 1995, 2000 by Stephen L. Moshier

  This software is derived from the Cephes Math Library and is
  incorporated herein by permission of the author.

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
      * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Abramowitz & Stegun polynomial approximations

The `_AS_*` coefficient tables in `lucid/_ops/composite/_bessel.py` come from
the *Handbook of Mathematical Functions* (Abramowitz & Stegun, National Bureau
of Standards, 1964), a work of the United States government; no notice is
required.
