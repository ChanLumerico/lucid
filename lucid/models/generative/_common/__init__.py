"""Shared substrate for the generative families.

What lives here is the code more than one family needs and none of them
owns: the config bases every family's config inherits, the diffusion
schedulers, the recurrent state-space trunk the world models share, and
the DiT block DiT and MeanFlow both build on.

The placement rule is R1 — a private module with one consumer stays
inside that family, and only a second real consumer moves it up.  This
sub-package is where "up" is.  Keeping it out of the domain's own
directory listing is the point: a reader opening
``lucid/models/generative/`` should see families, not the scaffolding
they stand on.

Nothing here is public.  Families import from it; it imports from no
family, which is what stops two siblings becoming coupled through a
module one of them happens to host.
"""
