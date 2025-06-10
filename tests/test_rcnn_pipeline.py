import pytest

np = pytest.importorskip("numpy")

import lucid
from lucid.models.conv.lenet import lenet_1
from lucid.models.conv.rcnn import RCNN


def test_rcnn_pipeline_runs():
    backbone = lenet_1()
    model = RCNN(
        backbone,
        feat_dim=10,
        num_classes=2,
        image_means=(0.0,),
        warper_output_size=(28, 28),
        add_one=False,
    )

    images = lucid.random.rand(2, 1, 32, 32)
    rois = [
        lucid.tensor([[0, 0, 15, 15], [8, 8, 31, 31]], dtype=lucid.Int32),
        lucid.tensor([[4, 4, 20, 20]], dtype=lucid.Int32),
    ]

    cls_scores, bbox_deltas = model(images, rois)
    num_rois = sum(len(r) for r in rois)
    assert cls_scores.shape == (num_rois, 2)
    assert bbox_deltas.shape == (num_rois, 2, 4)

    preds = model.predict(images, rois)
    assert len(preds) == 2
    for p in preds:
        assert set(p) == {"boxes", "scores", "labels"}
        assert p["boxes"].ndim == 2
        assert p["scores"].ndim == 1
        assert p["labels"].ndim == 1
