"""Shared substrate for the vision families.

For most of this domain's life there was nothing here, and the note in
[[arch-models-add-family]] said so approvingly: a family's private module
belongs inside that family until a *second* family actually needs it.
The JEPA pair is that second family.  I-JEPA and V-JEPA are the same
transformer over different tokens — patches of an image, tubelets of a
clip — so the block, its attention, its feed-forward and the position
arithmetic are shared, while the patch embedding, the position table's
rank and the masking stay with whichever family they describe.

Nothing here is public.  Families import from it; it imports from no
family, which is what stops two siblings becoming coupled through a
module one of them happens to host.
"""
