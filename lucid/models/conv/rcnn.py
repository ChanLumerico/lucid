import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


class RegionWarper(nn.Module):
    def __init__(self, output_size: tuple[int, int] = (224, 224)) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, images: Tensor, rois: list[Tensor]) -> Tensor:
        device = images.device
        B, C, H_img, W_img = images.shape

        M = sum(r.shape[0] for r in rois)
        if M == 0:
            return lucid.empty(0, C, *self.output_size, device=device)

        boxes = lucid.concatenate(rois, axis=0).to(device)
        img_ids = lucid.concatenate([...])  # NOTE: Implement `lucid.full`

        # TODO: Should implement `F.affine_grad` and `F.grid_sample`
