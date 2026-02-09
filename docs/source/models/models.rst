lucid.models
============

The `lucid.models` package provides a collection of predefined neural network 
architectures that are ready to use for various tasks, such as image classification 
and feature extraction. These models are designed to demonstrate key deep learning 
concepts while leveraging the modular and educational nature of the `lucid` framework.

Computer Vision
~~~~~~~~~~~~~~~

Computer Vision (CV) is a field of artificial intelligence that enables machines to interpret 
and understand visual information from the world, such as images and videos. It involves 
teaching computers to process, analyze, and make sense of visual data in a way similar to 
human vision.

.. list-table::
    :header-rows: 1
    :align: left

    * - Task
      - Description
      - Docs
    
    * - Image classification
      - Image classification is a key task in computer vision where a model assigns labels 
        to images based on their content. It processes the image through layers to extract 
        features and predict the most likely class.
      - `Image Classification <imgclf/index>`_
    
    * - Image generation
      - Image generation is a fundamental task in generative modeling where a model learns 
        to create new images that resemble a given dataset. It involves learning patterns 
        and structures from data to synthesize realistic or novel visuals.
      - `Image Generation <imggen/index>`_
    
    * - Object Detection
      - Object detection is a computer vision task that identifies and classifies multiple 
        objects within an image. It assigns labels and draws bounding boxes around each 
        detected object, combining localization with classification.
      - `Object Detection <objdet/index>`_

Natural Language Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Natural Language Processing (NLP) is a field of artificial intelligence that enables computers to 
understand, interpret, and generate human language. It combines linguistics and machine learning 
to process text or speech, allowing models to perform tasks like translation, summarization, 
and sentiment analysis.

.. list-table::
    :header-rows: 1
    :align: left

    * - Task
      - Description
      - Docs
    
    * - Sequence-to-Sequence
      - A sequence-to-sequence model is a type of neural network architecture used to transform 
        one sequence into another, such as translating a sentence from one language to another.
      - `Sequence-to-Sequence <seq2seq/index>`_

    * - Sequence Classification
      - A sequence classification model is a type of neural network architecture used to assign
        a label to an input sequence, such as determining sentiment or topic from a sentence.
      - `Sequence Classification <seqclf/index>`_
