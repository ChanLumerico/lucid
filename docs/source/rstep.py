MISC_EP = """
.. |wip-badge| raw:: html

    <span class="badge wip">Work-In-Progress</span>
"""

ARCH_EP = """
.. |convnet-badge| raw:: html

    <span class="badge convnet">ConvNet</span>

.. |one-stage-det-badge| raw:: html

    <span class="badge one_stage_det">One-Stage Detector</span>

.. |two-stage-det-badge| raw:: html

    <span class="badge two_stage_det">Two-Stage Detector</span>

.. |transformer-badge| raw:: html

    <span class="badge transformer">Transformer</span>

.. |vision-transformer-badge| raw:: html

    <span class="badge vision_transformer">Vision Transformer</span>

.. |detection-transformer-badge| raw:: html

    <span class="badge detection_transformer">Detection Transformer</span>

.. |encoder-only-transformer-badge| raw:: html

    <span class="badge encoder_only_transformer">Encoder-Only Transformer</span>

.. |autoencoder-badge| raw:: html

    <span class="badge autoencoder">Autoencoder</span>

.. |vae-badge| raw:: html

    <span class="badge vae">Variational Autoencoder</span>

.. |diffusion-badge| raw:: html

    <span class="badge diffusion">Diffusion</span>

.. |score-diffusion-badge| raw:: html

    <span class="badge score_based_diffusion">Score-Based Diffusion</span>
"""

BACKEND_EP = """
.. |cpp-badge| raw:: html

    <span class="badge normal">C++ Backend</span>
"""


def get_total_epilogs() -> str:
    return MISC_EP + ARCH_EP + BACKEND_EP
