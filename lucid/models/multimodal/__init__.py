"""Models whose input spans more than one modality.

The domain exists because the alternative is a lie about where a model
belongs.  A CLIP filed under ``vision`` would hide a text Transformer
that is half its parameters and all of its transfer story; filed under
``text`` it would hide the image tower.  The zoo classifies by
structure, and a two-tower contrastive model has a structure neither
single-modality domain describes.
"""
