Image Generation
================

.. toctree::
    :maxdepth: 1
    :hidden:

    VAE <autoencoder/VAE.rst>
    DDPM <diffusion/DDPM.rst>

Variational Autoencoder (VAE)
-----------------------------
|autoencoder-badge| |vae-badge| |imggen-badge|

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
      - `VAE <autoencoder/VAE>`_
      - :math:`(N,C,H,W)`

DDPM
----
|diffusion-badge| |imggen-badge|

A Denoising Diffusion Probabilistic Model (DDPM) is a generative model that learns to 
generate data by reversing a gradual noising process. It adds Gaussian noise over 
several timesteps and trains a neural network to denoise and recover the original 
data distribution through a Markovian process.

 J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,”  
 *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape

    * - DDPM
      - `DDPM <diffusion/DDPM>`_
      - :math:`(N,C,H,W)`

*To be implemented...🔮*
