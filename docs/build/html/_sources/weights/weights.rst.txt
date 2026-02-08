lucid.weights
=============

Overview
--------
The :mod:`lucid.weights` package provides **versioned, named references to 
pre-trained model parameters**. Each model family exposes an `Enum` class 
whose members describe a concrete checkpoint 
(e.g. training dataset, recipe, and preprocessing). 

Passing one of these `Enum` members to a model factory via the `weights=` 
parameter will **download (or resolve) the state dict** and **load** 
it into the model for you.

.. tip::

    Think of a weight Enum as a *pointer with metadata* to a specific set of
    parameters and its matching preprocessing.

Typical API Surface
-------------------
Every weight Enum follows a common shape:

- **Members**: named variants (e.g. `IMAGENET1K`, `DEFAULT`).
- **Attributes** (per member):

  - `name`: canonical tag for the checkpoint.
  - `meta`: dictionary-like metadata 
    (e.g. `num_classes`, `acc@1`, `acc@5`, `categories`).

.. note::

    Exact field names may differ slightly between model families, but **all weight
    Enums are loadable** through the `weights=` argument.

How Model Factories Use Weights
-------------------------------

Model generator functions accept a `weights` argument. When provided with a
member of the corresponding weights `Enum`, Lucid will:

1. **Instantiate** the architecture with canonical hyper-parameters.
2. **Fetch/resolve** the checkpoint (e.g. from a local cache or remote URL).
3. **Load** the returned state dict into the model via
   :py:meth:`lucid.nn.Module.load_state_dict`.

4. **Return** the model instance, ready for inference or further finetuning.

.. warning::

    Loading is done **strictly by default** (matching all parameter names/shapes).
    If you alter the head (e.g. different `num_classes`), call
    :py:meth:`lucid.nn.Module.load_state_dict` yourself with `strict=False` and
    a compatible state dict.

Quickstart Examples
-------------------

ResNet-18 (ImageNet-1k) with preprocessing:

.. code-block:: python

    import lucid
    import lucid.nn as nn
    from lucid.models.imgclf.resnet import resnet_18
    from lucid.weights import ResNet_18_Weights

    # Build model with pretrained weights
    model = resnet_18(weights=ResNet_18_Weights.IMAGENET1K)
    model.eval()

    # Example input (N, C, H, W) -> preprocess -> forward
    x = lucid.randn(1, 3, 224, 224)
    y = model(x)

Finetuning from a pretrained checkpoint while replacing the head:

.. code-block:: python

    from lucid.models.imgclf.resnet import resnet_50
    from lucid.weights import ResNet_50_Weights

    model = resnet_50(weights=ResNet_50_Weights.IMAGENET1K)

    # Replace the classification head for a custom 10-class task
    model.fc = nn.Linear(model.fc.in_features, 10)

    # Reload, allowing missing/unexpected keys (head differs)
    sd = ResNet_50_Weights.IMAGENET1K.state_dict()
    model.load_state_dict(sd, strict=False)

    # Now train on your dataset

Accessing metadata & categories
-------------------------------

.. code-block:: python

    w = ResNet_18_Weights.IMAGENET1K_V1
    print(w.name)          # 'IMAGENET1K_V1'
    print(w.meta["acc@1"])  # e.g. 69.8

Device & caching
----------------

- The returned model obeys :py:meth:`lucid.nn.Module.to` for device moves (`"cpu"` ⇄ `"gpu"`).
- Weight resolution **should** leverage a local cache when available; if a download
  is required, it happens during the `weights` load step.

.. tip::

    After loading, you can always re-save the state dict with `model.state_dict()`
    and manage it manually in your own experiment flow.

Custom / user-provided weights
------------------------------
If you have your own checkpoint:

- **Direct load** via `model.load_state_dict(my_state_dict)`.
- Or **wrap** it by using the family’s weight Enum conventions in your codebase so
  you can still pass `weights=YourWeights.MY_CHECKPOINT` to factory functions.

Common Pitfalls
---------------

- **Mismatched shapes** when changing classifier heads → use `strict=False`.
- **Input resolution** must match the training recipe (e.g. `224×224` for many
  ImageNet models).

Each resides under :mod:`lucid.weights` and is discoverable via auto-complete.
