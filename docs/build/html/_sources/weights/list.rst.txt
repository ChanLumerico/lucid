Pre-Trained Weights
===================

LeNet
-----
|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.weights.LeNet_1_Weights
.. autoclass:: lucid.weights.LeNet_4_Weights
.. autoclass:: lucid.weights.LeNet_5_Weights

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size
      - Metric (acc@1)
    
    * - *LeNet_1_Weights*
      - MNIST
      - mnist
      - 3,246
      - 13.4 KB
      - 0.7937
    
    * - *LeNet_4_Weights*
      - MNIST
      - mnist
      - 18,378
      - 74.1 KB
      - 0.8210
    
    * - *LeNet_5_Weights*
      - MNIST
      - mnist
      - 61,706
      - 248 KB
      - 0.9981

AlexNet
-------
|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.weights.AlexNet_Weights

.. list-table::
    :header-rows: 1
    :align: left

    * - Enum Name
      - Tag
      - Dataset
      - Parameter Size
      - File Size
      - Metric (acc@1)
      - Metric (acc@5)
    
    * - *AlexNet_Weights*
      - IMAGENET1K
      - imagenet_1k
      - 61,100,840
      - 244 MB
      - 0.5652
      - 0.7906
