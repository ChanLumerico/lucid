from lucid import register_model

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.models.imgclf import vggnet_16


__all__ = ["YOLO_V2"]


class YOLO_V2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_anchors: int = 5,
        anchors: list[tuple[float, float]] | None = None,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        darknet: nn.Module | None = None,
        route_layer: int | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        if anchors is None:
            anchors = [
                (1.3221, 1.73145),
                (3.19275, 4.00944),
                (5.05587, 8.09892),
                (9.47112, 4.84053),
                (11.2364, 10.0071),
            ]
        self.anchors = Tensor(anchors, dtype=lucid.Float32)

        self.darknet = darknet if darknet is not None else vggnet_16().conv[:30]
        self.route_layer = route_layer if route_layer is not None else 22

        route_c = self._layer_out_channels(self.darknet, self.route_layer)
        out_c = self._darknet_out_channels(self.darknet)

        self.passthrough_conv = nn.Sequential(
            nn.Conv2d(route_c, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )

        self.head_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head_conv1 = nn.Sequential(
            nn.Conv2d(out_c, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.head_conv2 = nn.Sequential(
            nn.Conv2d(1024 + 256, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        self.detector = nn.Conv2d(1024, num_anchors * (5 + num_classes), kernel_size=1)

    @staticmethod
    def _darknet_out_channels(darknet: nn.Module) -> int:
        out_c: int | None = None
        for module in darknet.modules():
            if isinstance(module, nn.Conv2d):
                out_c = module.out_channels
        if out_c is None:
            raise ValueError("No Conv2d layer found in darknet")
        return out_c

    @staticmethod
    def _layer_out_channels(darknet: nn.Module, idx: int) -> int:
        out_c: int | None = None
        for i, module in enumerate(darknet):
            if isinstance(module, nn.Conv2d):
                out_c = module.out_channels
            if i == idx:
                break
        if out_c is None:
            raise ValueError("No Conv2d layer found in darknet")
        return out_c

    @staticmethod
    def _reorg(x: Tensor, stride: int = 2) -> Tensor:
        n, c, h, w = x.shape
        assert h % stride == 0 and w % stride == 0

        x = x.reshape(n, c, h // stride, stride, w // stride, stride)
        x = x.transpose((0, 1, 3, 5, 2, 4))
        return x.reshape(n, c * stride**2, h // stride, w // stride)

    def forward(self, x: Tensor) -> Tensor:
        passthrough: Tensor | None = None
        for i, layer in enumerate(self.darknet):
            x = layer(x)
            if i == self.route_layer:
                passthrough = x

        assert passthrough is not None

        x = self.head_pool(x)
        x = self.head_conv1(x)

        p = self.passthrough_conv(passthrough)
        p = self._reorg(p)

        x = lucid.concatenate([p, x], axis=1)
        x = self.head_conv2(x)
        x = self.detector(x)

        N, _, H, W = x.shape
        x = x.transpose((0, 2, 3, 1))
        return x.reshape(N, H, W, self.num_anchors * (5 + self.num_classes))

    def get_loss(self, x: Tensor, target: Tensor) -> Tensor: ...
