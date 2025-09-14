from lucid import register_model

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import _DeviceType

from lucid.models.imgclf.cspnet import csp_darknet_53
from lucid.models.objdet.util import DetectionDict


__all__ = ["YOLO_V4", "yolo_v4"]


DEFAULT_ANCHORS: list[list[tuple[int, int]]] = [
    [(12, 16), (19, 36), (40, 28)],
    [(36, 75), (76, 55), (72, 146)],
    [(142, 110), (192, 243), (459, 401)],
]
DEFAULT_STRIDES: list[int] = [8, 16, 32]


def _to_xyxy(xywh: Tensor) -> tuple[Tensor, Tensor]:
    c = xywh[..., :2]
    wh = xywh[..., 2:]
    x1y1 = c - wh * 0.5
    x2y2 = c + wh * 0.5
    return x1y1, x2y2


def _bbox_iou_xywh(pred_xywh: Tensor, tgt_xywh: Tensor, eps: float = 1e-7) -> Tensor:
    p1, p2 = _to_xyxy(pred_xywh)
    t1, t2 = _to_xyxy(tgt_xywh)

    inter_lt = lucid.maximum(p1, t1)
    inter_rb = lucid.minimum(p2, t2)
    inter_wh = (inter_rb - inter_lt).clip(0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    area_p = (p2[..., 0] - p1[..., 0]).clip(0) * (p2[..., 1] - p1[..., 1]).clip(0)
    area_t = (t2[..., 0] - t1[..., 0]).clip(0) * (t2[..., 1] - t1[..., 1]).clip(0)
    union = area_p + area_t - inter + eps
    return inter / union


def _bbox_iou_ciou(pred_xywh: Tensor, tgt_xywh: Tensor, eps: float = 1e-7) -> Tensor:
    p1, p2 = _to_xyxy(pred_xywh)
    t1, t2 = _to_xyxy(tgt_xywh)
    iou = _bbox_iou_xywh(pred_xywh, tgt_xywh, eps)

    enc_lt = lucid.minimum(p1, t1)
    enc_rb = lucid.maximum(p2, t2)
    enc_wh = (enc_rb - enc_lt).clip(0)
    c2 = enc_wh[..., 0] ** 2 + enc_wh[..., 1] ** 2 + eps

    rho2 = (pred_xywh[..., 0] - tgt_xywh[..., 0]) ** 2
    rho2 += (pred_xywh[..., 1] - tgt_xywh[..., 1]) ** 2

    v = (4 / lucid.pi**2) * (
        lucid.arctan(tgt_xywh[..., 2] / (tgt_xywh[..., 3] + eps))
        - lucid.arctan(pred_xywh[..., 2] / (pred_xywh[..., 3] + eps))
    ) ** 2

    with lucid.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2) - alpha * v


def _iou_xyxy(a: Tensor, b: Tensor, eps: float = 1e-7) -> Tensor:
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)

    tl = lucid.maximum(a[:, None, :2], b[:, :2])
    br = lucid.minimum(a[:, None, 2:], b[:, 2:])
    wh = (br - tl).clip(0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    return inter / (area_a[:, None] + area_b - inter + eps)


def _diou_xyxy(a: Tensor, b: Tensor, eps: float = 1e-7) -> Tensor:
    iou = _iou_xyxy(a, b)
    ac = (a[:, :2] + a[:, 2:]) * 0.5
    bc = (b[:, :2] + b[:, 2:]) * 0.5

    diff = ac[:, None, :] - bc[None, :, :]
    rho2 = (diff**2).sum(axis=-1)

    enc_tl = lucid.minimum(a[:, None, :2], b[:, :2])
    enc_br = lucid.maximum(a[:, None, 2:], b[:, 2:])
    enc_wh = (enc_br - enc_tl).clip(0)

    c2 = (enc_wh**2).sum(axis=-1) + eps
    return iou - (rho2 / c2)


def _smooth_bce(eps: float = 0.0) -> tuple[float, float]:
    return 1.0 - 0.5 * eps, 0.5 * eps


class _DefaultCSPDarkNet53(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = csp_darknet_53(act=nn.Mish)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        outs = self.net.forward_features(x, return_stage_out=True)
        H, W = x.shape[-2:]

        chosen: dict[int, Tensor] = {}
        for f in outs:
            fh, fw = f.shape[-2], f.shape[-1]

            sh = int(round(H / max(fh, 1)))
            sw = int(round(W / max(fw, 1)))
            s = max(sh, sw)
            if s in (8, 16, 32):
                chosen[s] = f

        return chosen[8], chosen[16], chosen[32]


class _ConvBNAct(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        act: bool = True,
    ) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.LeakyReLU(0.1) if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class _SPPBlock(nn.Module):
    def __init__(self, c_in: int, c_mid: int | None = None) -> None:
        super().__init__()
        c_mid = c_mid if c_mid is not None else c_in // 2
        self.conv1 = _ConvBNAct(c_in, c_mid, k=1)
        self.conv2 = _ConvBNAct(c_mid, c_in, k=3)
        self.conv3 = _ConvBNAct(c_in, c_mid, k=1)

        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in (5, 9, 13)]
        )

        self.conv4 = _ConvBNAct(c_mid * 4, c_mid, k=1)
        self.conv5 = _ConvBNAct(c_mid, c_in, k=3)
        self.conv6 = _ConvBNAct(c_in, c_mid, k=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        pooled = [x] + [pool(x) for pool in self.pools]
        x = lucid.concatenate(pooled, axis=1)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class _FiveConv(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            _ConvBNAct(in_ch, mid_ch, k=1),
            _ConvBNAct(mid_ch, mid_ch * 2, k=3),
            _ConvBNAct(mid_ch * 2, mid_ch, k=1),
            _ConvBNAct(mid_ch, mid_ch * 2, k=3),
            _ConvBNAct(mid_ch * 2, out_ch, k=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class _PANetNeck(nn.Module):
    def __init__(
        self,
        in_channels: list[int] = (256, 512, 1024),
        out_channels: list[int] = (256, 512, 1024),
    ) -> None:
        super().__init__()
        c3, c4, c5 = in_channels
        o3, o4, o5 = out_channels
        assert c3 == o3 and c4 == o4 and c5 == o5

        self.spp = _SPPBlock(c5)

        self.reduce_c5 = _ConvBNAct(c5 // 2, c4, k=1)
        self.c4_lat = _ConvBNAct(c4, c4, k=1)
        self.c4_td = _FiveConv(c4 * 2, c4, c4)

        self.reduce_c4 = _ConvBNAct(c4, c3, k=1)
        self.c3_lat = _ConvBNAct(c3, c3, k=1)
        self.c3_out = _FiveConv(c3 * 2, c3, c3)

        self.down_c3 = _ConvBNAct(o3, c4, k=3, s=2)
        self.c4_out = _FiveConv(c4 * 2, c4, o4)

        self.down_c4 = _ConvBNAct(o4, c4, k=3, s=2)
        self.c5_out = _FiveConv(c4 + c5 // 2, c5 // 2, o5)

    @staticmethod
    def _upsample(x: Tensor, ref: Tensor) -> Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    @staticmethod
    def _match_hw(x: Tensor, ref: Tensor) -> Tensor:
        return (
            x
            if x.shape[-2:] == ref.shape[-2:]
            else F.interpolate(x, size=ref.shape[-2:], mode="nearest")
        )

    def forward(self, feats: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        c3, c4, c5 = feats
        p5_spp = self.spp(c5)

        u5 = self._upsample(self.reduce_c5(p5_spp), c4)
        p4_td = self.c4_td(lucid.concatenate([self.c4_lat(c4), u5], axis=1))

        u4 = self._upsample(self.reduce_c4(p4_td), c3)
        p3 = self.c3_out(lucid.concatenate([self.c3_lat(c3), u4], axis=1))

        d3 = self._match_hw(self.down_c3(p3), p4_td)
        p4 = self.c4_out(lucid.concatenate([d3, p4_td], axis=1))

        d4 = self._match_hw(self.down_c4(p4), p5_spp)
        p5 = self.c5_out(lucid.concatenate([d4, p5_spp], axis=1))

        return p3, p4, p5


class _YOLOHead(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int, int],
        num_anchors: int,
        num_classes: int,
        iou_aware_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.detect = nn.ModuleList()

        out_per_anchor = 6 + num_classes + (1 if iou_aware_alpha > 0 else 0)
        for c in in_channels:
            self.detect.append(
                nn.Sequential(
                    _ConvBNAct(c, c, k=3),
                    nn.Conv2d(c, num_anchors * out_per_anchor, kernel_size=1),
                )
            )

    def forward(self, feats: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        return tuple(self.detect[i](f) for i, f in enumerate(feats))


class YOLO_V4(nn.Module):
    def __init__(
        self,
        num_classes: int,
        anchors: list[list[tuple[int, int]]] | None = None,
        strides: list[int] | None = None,
        backbone: nn.Module | None = None,
        backbone_out_channels: tuple[int, int, int] | None = None,
        in_channels: tuple[int, int, int] = (256, 512, 1024),
        pos_iou_thr: float = 0.25,
        ignore_iou_thr: float = 0.5,
        obj_balance: tuple[float, float, float] = (4.0, 1.0, 0.4),
        cls_label_smoothing: float = 0.0,
        iou_aware_alpha: float = 0.5,
        iou_branch_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors if anchors is not None else DEFAULT_ANCHORS
        self.strides = strides if strides is not None else DEFAULT_STRIDES

        self.backbone = backbone if backbone is not None else _DefaultCSPDarkNet53()
        self.neck = _PANetNeck(in_channels, in_channels)
        self.head = _YOLOHead(in_channels, len(self.anchors[0]), num_classes)

        if backbone is None:
            assert backbone_out_channels is None
            backbone_out_channels = (128, 256, 512)
        else:
            assert backbone_out_channels is not None

        self.backbone_out_channels = backbone_out_channels
        self.in_channels = in_channels

        self.c3_conv = _ConvBNAct(backbone_out_channels[0], in_channels[0], k=1)
        self.c4_conv = _ConvBNAct(backbone_out_channels[1], in_channels[1], k=1)
        self.c5_conv = _ConvBNAct(backbone_out_channels[2], in_channels[2], k=1)

        self.pos_iou_thr = pos_iou_thr
        self.ignore_iou_thr = ignore_iou_thr
        self.obj_balance = obj_balance
        self.cls_eps = cls_label_smoothing

        self.iou_aware_alpha = iou_aware_alpha
        self.iou_branch_weight = iou_branch_weight

        self._bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        c3, c4, c5 = self.backbone(x)
        if (c3.shape[1], c4.shape[1], c5.shape[1]) != self.in_channels:
            c3, c4, c5 = (
                self.c3_conv(c3),
                self.c4_conv(c4),
                self.c5_conv(c5),
            )

        p3, p4, p5 = self.neck((c3, c4, c5))
        outs = self.head((p3, p4, p5))
        return outs

    def _decode_outputs(
        self, preds: tuple[Tensor, ...], img_size: tuple[int, int]
    ) -> tuple[Tensor, ...]:
        device = preds[0].device
        B = preds[0].shape[0]
        outputs = [[] for _ in range(B)]

        for s, p in enumerate(preds):
            A = len(self.anchors[s])
            C = self.num_classes
            B_, _, H, W = p.shape
            assert B == B_

            p = p.reshape(B, A, 6 + C, H, W).transpose((0, 1, 3, 4, 2))
            yv, xv = lucid.meshgrid(
                lucid.arange(H, device=device),
                lucid.arange(W, device=device),
                indexing="ij",
            )

            grid = lucid.stack([xv, yv], axis=2).reshape(1, 1, H, W, 2)
            anc = (
                lucid.tensor(self.anchors[s], device=device)
                .astype(lucid.Float)
                .reshape(1, A, 1, 1, 2)
            )
            stride = self.strides[s]

            xy = (F.sigmoid(p[..., 0:2]) + grid) * stride
            wh = F.exp(p[..., 2:4]) * anc
            objp = F.sigmoid(p[..., 4])
            ioup = F.sigmoid(p[..., 5])
            cl = F.sigmoid(p[..., 6:])

            alpha = self.iou_aware_alpha
            fused = (objp ** (1 - alpha)) * (ioup**alpha)

            x1y1 = xy - wh * 0.5
            x2y2 = xy + wh * 0.5
            x1y1[..., 0] = x1y1[..., 0].clip(0, img_size[1] - 1)
            x1y1[..., 1] = x1y1[..., 1].clip(0, img_size[0] - 1)
            x2y2[..., 0] = x2y2[..., 0].clip(0, img_size[1] - 1)
            x2y2[..., 1] = x2y2[..., 1].clip(0, img_size[0] - 1)

            out = lucid.concatenate([x1y1, x2y2, fused.unsqueeze(-1), cl], axis=-1)
            out = out.reshape(B, -1, 5 + C)

            for b in range(B):
                outputs[b].append(out[b])

        return tuple(lucid.concatenate(o, axis=0) for o in outputs)

    @lucid.no_grad()
    def _diou_nms_per_img(
        self, det: Tensor, conf_thresh: float, diou_thresh: float, max_det: int = 300
    ) -> Tensor:
        if det.size == 0:
            return lucid.zeros(0, 6, requires_grad=det.requires_grad, device=det.device)

        obj, cl = det[:, 4:5], det[:, 5:]
        conf, cl_id = (obj * cl).max(axis=1)
        mask = conf > conf_thresh
        if mask.sum() == 0:
            return lucid.zeros(0, 6, requires_grad=det.requires_grad, device=det.device)

        boxes = det[mask, :4]
        conf = conf[mask]
        cl_id = cl_id[mask].astype(lucid.Int)

        keep_idx = []
        for c in lucid.unique(cl_id):
            idx = lucid.nonzero(cl_id == c).squeeze(axis=1)
            b = boxes[idx]
            s = conf[idx]

            order = lucid.argsort(s, descending=True)
            idx = idx[order]
            b = b[order]

            while idx.size > 0:
                i = idx[0]
                keep_idx.append(i.item())
                if idx.size == 1:
                    break

                diou = _diou_xyxy(b[:1], b[1:]).squeeze(axis=0)
                mask_ = diou <= diou_thresh
                idx = idx[1:][mask_]
                b = b[1:][mask_]

        keep = lucid.tensor(keep_idx, device=det.device, dtype=lucid.Int32)[:max_det]
        return lucid.concatenate(
            [boxes[keep], conf[keep, None], cl_id[keep, None]], axis=1
        )

    def _build_targets(
        self,
        targets: list[Tensor],
        feat_shapes: list[tuple[int, int]],
        device: _DeviceType,
    ) -> tuple[list[Tensor], ...]:
        B = len(targets)
        tcls, tbox, tindices, tobj, ignore = [], [], [], [], []
        for s in range(3):
            na = len(self.anchors[s])
            H, W = feat_shapes[s]

            tcls.append([])
            tbox.append([])
            tindices.append([])

            tobj.append(lucid.zeros(B, na, H, W, device=device))
            ignore.append(lucid.zeros(B, na, H, W, dtype=bool, device=device))

        all_anc = lucid.tensor(
            [a for sc in self.anchors for a in sc], device=device, dtype=lucid.Float32
        )

        for b, tgt in enumerate(targets):
            if tgt.size == 0:
                continue

            cl = tgt[:, 0].astype(lucid.Int32)
            boxes = tgt[:, 1:5].astype(lucid.Float)

            gtw, gth = boxes[:, 2], boxes[:, 3]
            inter = lucid.minimum(gtw[:, None], all_anc[None, :, 0])
            inter *= lucid.minimum(gth[:, None], all_anc[None, :, 1])
            union = (
                (gtw[:, None] * gth[:, None])
                + (all_anc[None, :, 0] * all_anc[None, :, 1])
                - inter
                + 1e-9
            )
            wh_iou = inter / union

            pos_mask_all = wh_iou >= self.pos_iou_thr
            best_idx = lucid.argmax(wh_iou, axis=1, keepdims=True)
            pos_mask_all = lucid.where(
                pos_mask_all.any(axis=1, keepdims=True),
                pos_mask_all,
                F.one_hot(best_idx.squeeze(axis=1), wh_iou.shape[1], dtype=bool),
            )

            for i in range(boxes.shape[0]):
                gx, gy, gw, gh = boxes[i]
                gi_s, gj_s = [], []
                for s in range(3):
                    stride = self.strides[s]
                    gi_s.append(
                        int((gx / stride).clip(0, feat_shapes[s][1] - 1).item())
                    )
                    gj_s.append(
                        int((gy / stride).clip(0, feat_shapes[s][0] - 1).item())
                    )

                for a_global in lucid.nonzero(pos_mask_all[i]).flatten().tolist():
                    s = a_global // 3
                    a_local = a_global % 3
                    gi, gj = gi_s[s], gj_s[s]

                    tobj[s][b, a_local, gj, gi] = 1.0
                    tbox[s].append(lucid.tensor([gx, gy, gw, gh], device=device))
                    tcls[s].append(cl[i])
                    tindices[s].append((b, a_local, gj, gi))

                for a_global in range(9):
                    s = a_global // 3
                    a_local = a_global % 3
                    if (
                        wh_iou[i, a_global] >= self.ignore_iou_thr
                        and not pos_mask_all[i, a_global]
                    ):
                        gi, gj = gi_s[s], gj_s[s]
                        ignore[s][b, a_local, gj, gi] = True

        for s in range(3):
            if len(tbox[s]) == 0:
                tbox[s] = lucid.zeros(0, 4, device=device)
                tcls[s] = lucid.zeros(0, dtype=lucid.Int32, device=device)
                tindices[s] = lucid.zeros(0, 4, dtype=lucid.Int32, device=device)
            else:
                tbox[s] = lucid.stack(tbox[s], axis=0)
                tcls[s] = lucid.stack(tcls[s], axis=0)
                tindices[s] = lucid.tensor(
                    tindices[s], dtype=lucid.Int32, device=device
                )

        return tcls, tbox, tindices, tobj, ignore

    def get_loss(self, x: Tensor, targets: list[Tensor]) -> Tensor:
        B, _, H, W = x.shape
        device = x.device
        preds = self.forward(x)

        C = self.num_classes
        feat_shapes = [(p.shape[2], p.shape[3]) for p in preds]

        tcls, tbox, tindices, tobj, ignore = self._build_targets(
            targets, feat_shapes, device
        )
        pos_t, neg_t = _smooth_bce(self.cls_eps)

        lbox = lucid.zeros(1, device=device)
        lobj = lucid.zeros(1, device=device)
        lcls = lucid.zeros(1, device=device)
        liou = lucid.zeros(1, device=device)

        img_area = float(H * W)
        for s, p in enumerate(preds):
            na = len(self.anchors[s])
            Hs, Ws = feat_shapes[s]

            p = p.reshape(B, na, 6 + C, Hs, Ws).transpose((0, 1, 3, 4, 2))
            obj_logit = p[..., 4]

            pos_mask = tobj[s] > 0
            neg_mask = (~pos_mask) & (~ignore[s])

            if pos_mask.any():
                lobj_pos = self._bce(
                    obj_logit[pos_mask], lucid.ones_like(obj_logit[pos_mask])
                )
                lobj += self.obj_balance[s] * lobj_pos

            if neg_mask.any():
                lobj_neg = self._bce(
                    obj_logit[neg_mask], lucid.zeros_like(obj_logit[neg_mask])
                )
                lobj += self.obj_balance[s] * lobj_neg

            if tindices[s].shape[0] == 0:
                continue
            b, a, gj, gi = (
                tindices[s][:, 0],
                tindices[s][:, 1],
                tindices[s][:, 2],
                tindices[s][:, 3],
            )
            ps = p[b, a, gj, gi]

            stride = self.strides[s]
            anc = lucid.tensor(self.anchors[s], device=device, dtype=lucid.Float32)[a]

            px = (F.sigmoid(ps[:, 0]) + gi.astype(lucid.Float32)) * stride
            py = (F.sigmoid(ps[:, 1]) + gj.astype(lucid.Float32)) * stride
            pw = F.exp(ps[:, 2]) * anc[:, 0]
            ph = F.exp(ps[:, 3]) * anc[:, 1]

            pred_xywh = lucid.stack([px, py, pw, ph], axis=1)

            gw, gh = tbox[s][:, 2], tbox[s][:, 3]
            box_w = 2.0 - (gw * gh) / max(img_area, 1.0)
            ciou = _bbox_iou_ciou(pred_xywh, tbox[s])
            lbox += (box_w * (1.0 - ciou)).mean()

            with lucid.no_grad():
                iou_target = _bbox_iou_xywh(pred_xywh, tbox[s]).clip(0.0, 1.0)
            liou += self._bce(ps[:, 5], iou_target) * self.iou_branch_weight

            if C > 1:
                t = lucid.full((ps.shape[0], C), neg_t, device=device)
                t[lucid.arange(ps.shape[0], device=device), tcls[s]] = pos_t
                lcls += F.binary_cross_entropy_with_logits(
                    ps[:, 6:], t, reduction="mean"
                )

        return lbox + lobj + lcls + liou

    @lucid.no_grad()
    def predict(
        self, x: Tensor, conf_thresh: float = 0.25, diou_thresh: float = 0.5
    ) -> list[list[DetectionDict]]:
        self.eval()
        H, W = x.shape[2:]
        preds = self.forward(x)
        decoded = self._decode_outputs(preds, (H, W))

        results: list[list[DetectionDict]] = []
        for det in decoded:
            kept = self._diou_nms_per_img(det, conf_thresh, diou_thresh)
            objs = [
                {
                    "box": row[:4].tolist(),
                    "score": row[4].item(),
                    "class_id": int(row[5].item()),
                }
                for row in kept
            ]
            results.append(objs)

        return results


@register_model
def yolo_v4(num_classes: int = 80, **kwargs) -> YOLO_V4:
    return YOLO_V4(
        num_classes=num_classes,
        pos_iou_thr=0.213,
        ignore_iou_thr=0.7,
        obj_balance=(1.0, 1.0, 1.0),
        cls_label_smoothing=0.0,
        iou_aware_alpha=0.0,
        iou_branch_weight=0.0,
        **kwargs,
    )
