"""Shared substrate for the multimodal families.

The config base the families inherit from.  Placement follows R1, the
same rule :mod:`lucid.models.generative._common` documents: a private
module with a single consumer stays in its family, and this
sub-package holds the ones with more.

Nothing here is public, and nothing here imports a family.
"""
