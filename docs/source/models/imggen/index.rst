Image Generation
================

.. toctree::
    :maxdepth: 1
    :hidden:

    VAE <vae/VAE.rst>

Variational Autoencoder (VAE)
-----------------------------
|autoencoder-badge| |imggen-badge|

A Variational Autoencoder (VAE) is a generative model that learns to encode input 
data into a probabilistic latent space and reconstruct it through a decoder. 
Unlike traditional autoencoders, VAEs model uncertainty by learning a distribution 
over latent variables, enabling smooth sampling and interpolation.

 D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” 
 *International Conference on Learning Representations (ICLR)*, 2014.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
    
    * - VAE
      - `VAE <vae/VAE>`_
      - :math:`(N,C,H,W)`

*To be implemented...🔮*
