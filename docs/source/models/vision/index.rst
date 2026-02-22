Vision Models
=============

.. toctree::
    :maxdepth: 1
    :hidden:

    Image Classification <imgclf/index.rst>
    Object Detection <objdet/index.rst>
    Image Segmentation <imgseg/index.rst>

.. list-table::
    :header-rows: 1
    :align: left

    * - Task
      - Explanation
      - Docs
    
    * - Image Classification
      - Assign one or more semantic labels to an input image by learning
        discriminative visual features. Typical outputs are a top-1 class
        prediction and class probabilities (top-k), and this task is commonly
        used for recognition benchmarks and as a pretrained backbone for
        downstream vision tasks.
      - `Image Classification <imgclf/index.rst>`_
    
    * - Object Detection
      - Detect and localize multiple objects in an image by predicting
        bounding boxes together with class labels and confidence scores for
        each instance. Unlike image classification, detection must answer both
        what is present and where it appears, making it suitable for scene
        understanding and real-world perception pipelines.
      - `Object Detection <objdet/index.rst>`_

    * - Image Segmentation
      - Perform dense visual understanding by assigning predictions at the
        pixel level. In instance segmentation, each object instance receives
        both a category label and its own binary mask, enabling fine-grained
        scene parsing beyond bounding boxes.
      - `Image Segmentation <imgseg/index.rst>`_
