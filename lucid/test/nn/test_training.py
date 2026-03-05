import importlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

import lucid
import lucid.nn as nn
import lucid.optim as optim
import lucid.optim.lr_scheduler as lr_scheduler

from lucid.test.core import ModuleTorchBase


class _TopLevelBackbone(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.identity = nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(
                num_features=8,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=False,
            ),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=8,
                        out_channels=8,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    nn.GroupNorm(num_groups=2, num_channels=8, eps=1e-5, affine=True),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=8,
                        out_channels=8,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    nn.GELU(),
                ),
            ]
        )
        self.layer_norm = nn.LayerNorm((8, 16, 16), eps=1e-5)
        self.dropout = nn.Dropout2d(p=0.0)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.flatten = nn.Flatten(start_axis=1, end_axis=-1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.LayerNorm(64, eps=1e-5),
            nn.Dropout(p=0.0),
            nn.Linear(in_features=64, out_features=num_classes, bias=True),
        )

    def forward(self, input_: Any) -> Any:
        x = self.identity(input_)
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.classifier(x)

        return x


class _TorchTopLevelBackbone(torch.nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.identity = torch.nn.Identity()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            torch.nn.BatchNorm2d(
                num_features=8,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=False,
            ),
            torch.nn.ReLU(),
        )
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=8,
                        out_channels=8,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    torch.nn.GroupNorm(num_groups=2, num_channels=8, eps=1e-5),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=8,
                        out_channels=8,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    torch.nn.GELU(approximate="tanh"),
                ),
            ]
        )
        self.layer_norm = torch.nn.LayerNorm((8, 16, 16), eps=1e-5)
        self.dropout = torch.nn.Dropout2d(p=0.0)
        self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=64, bias=True),
            torch.nn.LayerNorm(64, eps=1e-5),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(in_features=64, out_features=num_classes, bias=True),
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        x = self.identity(input_)
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.classifier(x)

        return x


class TestTopLevelModuleTraining(ModuleTorchBase):
    random_seed: int | None = 20260305
    batch_size: int = 8
    num_classes: int = 10
    steps: int = 64
    lr: float = 5e-3

    torch_seed_offset: int = 101
    scheduler_step_size: int = 2
    scheduler_gamma: float = 0.5
    compare_scheduler: bool = True
    scheduler_last_epoch: int = -1

    np_dtype: Any = np.float32
    rtol: float = 5e-5
    atol: float = 1e-6
    param_rtol: float = 5e-4
    param_atol: float = 1e-5
    param_grad_skip_threshold: float = 1e-7

    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu", "gpu")

    @staticmethod
    def _torch_device(device: str) -> str:
        return "mps" if device == "gpu" else device

    def _set_seeds(self, seed: int, device: str) -> None:
        random = importlib.import_module("lucid.random")
        random.seed(seed)

        torch_seed = seed + self.torch_seed_offset
        torch.manual_seed(torch_seed)
        torch_device = self._torch_device(device)

        if torch_device == "mps" and hasattr(torch, "mps"):
            torch.mps.manual_seed(torch_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

    @staticmethod
    def _sync_model_state(
        lucid_model: nn.Module,
        torch_model: torch.nn.Module,
    ) -> None:
        lucid_params = tuple(lucid_model.parameters())
        torch_params = tuple(torch_model.parameters())

        if len(lucid_params) != len(torch_params):
            raise AssertionError(
                "Parameter count mismatch between lucid and torch models: "
                f"lucid={len(lucid_params)} torch={len(torch_params)}"
            )
        for lucid_param, torch_param in zip(lucid_params, torch_params):
            torch_param.data = torch.as_tensor(
                np.array(lucid_param.data),
                device=torch_param.device,
                dtype=torch_param.dtype,
            )

        lucid_buffers = tuple(lucid_model.buffers())
        torch_buffers = tuple(torch_model.buffers())

        if len(lucid_buffers) != len(torch_buffers):
            raise AssertionError(
                "Buffer count mismatch between lucid and torch models: "
                f"lucid={len(lucid_buffers)} torch={len(torch_buffers)}"
            )

        for lucid_buffer, torch_buffer in zip(lucid_buffers, torch_buffers):
            if lucid_buffer is None and torch_buffer is None:
                continue

            if lucid_buffer is None or torch_buffer is None:
                raise AssertionError("Buffer presence mismatch between models.")

            torch_buffer.data = torch.as_tensor(
                np.array(lucid_buffer.data),
                device=torch_buffer.device,
                dtype=torch_buffer.dtype,
            )

    @staticmethod
    def _iter_param_pairs(
        lucid_model: nn.Module,
        torch_model: torch.nn.Module,
    ) -> Iterator[tuple[Any, Any]]:
        lucid_params = tuple(lucid_model.parameters())
        torch_params = tuple(torch_model.parameters())

        if len(lucid_params) != len(torch_params):
            raise AssertionError(
                "Parameter count mismatch between lucid and torch models: "
                f"lucid={len(lucid_params)} torch={len(torch_params)}"
            )

        for lucid_param, torch_param in zip(lucid_params, torch_params):
            yield lucid_param, torch_param

    def _build_data(self, device: str) -> tuple[Any, Any, torch.Tensor, torch.Tensor]:
        rng = np.random.RandomState(self.random_seed + self.torch_seed_offset)
        x_np = rng.standard_normal((self.batch_size, 3, 32, 32)).astype(self.np_dtype)
        y_np = rng.standard_normal((self.batch_size, self.num_classes)).astype(
            self.np_dtype
        )

        torch_device = self._torch_device(device)
        x_torch = torch.tensor(
            x_np,
            dtype=torch.float32,
            device=torch_device,
            requires_grad=False,
        )
        y_torch = torch.tensor(
            y_np,
            dtype=torch.float32,
            device=torch_device,
            requires_grad=False,
        )

        x_lucid = self.tensor(
            x_np, requires_grad=False, dtype=lucid.Float32, device=device
        )
        y_lucid = self.tensor(
            y_np, requires_grad=False, dtype=lucid.Float32, device=device
        )

        return x_lucid, y_lucid, x_torch, y_torch

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()

        if hasattr(value, "data"):
            return np.array(value.data)

        return np.array(value)

    def test_module_op_forward_backward(self) -> None:
        return

    def test_modules_training_stepwise_match_torch(self) -> None:
        for device in self.tensor_op_devices():
            if not self._is_device_supported(device):
                continue

            self._set_seeds(seed=int(self.random_seed), device=device)
            x_lucid, y_lucid, x_torch, y_torch = self._build_data(device=device)

            lucid_model = _TopLevelBackbone(num_classes=self.num_classes)
            torch_model = _TorchTopLevelBackbone(num_classes=self.num_classes).to(
                self._torch_device(device)
            )
            lucid_model.to(device)

            self._sync_model_state(lucid_model, torch_model)
            lucid_model.train(True)
            torch_model.train(True)

            lucid_loss_fn = nn.MSELoss(reduction="mean")
            torch_loss_fn = torch.nn.MSELoss(reduction="mean")

            lucid_optimizer = optim.Adam(
                lucid_model.parameters(),
                lr=self.lr,
            )
            torch_optimizer = torch.optim.Adam(
                torch_model.parameters(),
                lr=self.lr,
            )
            lucid_scheduler = lr_scheduler.StepLR(
                optimizer=lucid_optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
                last_epoch=self.scheduler_last_epoch,
            )

            for group in torch_optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])

            torch_scheduler = torch.optim.lr_scheduler.StepLR(
                torch_optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
                last_epoch=self.scheduler_last_epoch,
            )

            for step in range(self.steps):
                lucid_optimizer.zero_grad()
                torch_optimizer.zero_grad()

                lucid_out = lucid_model(x_lucid)
                torch_out = torch_model(x_torch)

                assert_has = f"[step={step}, device={device}]"
                self.assert_tensor_allclose(
                    lucid_out,
                    torch_out,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                lucid_loss = lucid_loss_fn(lucid_out, y_lucid)
                torch_loss = torch_loss_fn(torch_out, y_torch)

                self.assert_tensor_allclose(
                    lucid_loss,
                    torch_loss,
                    rtol=self.rtol,
                    atol=self.atol,
                )

                lucid_loss.backward()
                torch_loss.backward()

                lucid_param_grads: list[Any] = []
                torch_param_grads: list[Any] = []
                for lucid_param, torch_param in self._iter_param_pairs(
                    lucid_model, torch_model
                ):
                    if lucid_param.grad is None:
                        assert torch_param.grad is None, (
                            f"grad mismatch on {assert_has}: "
                            "torch has grad while lucid does not."
                        )
                        continue

                    assert (
                        torch_param.grad is not None
                    ), f"missing torch grad on {assert_has}"

                    self.assert_tensor_allclose(
                        lucid_param.grad,
                        torch_param.grad.detach().cpu().numpy(),
                        rtol=self.param_rtol,
                        atol=self.param_atol,
                    )
                    lucid_param_grads.append(lucid_param.grad)
                    torch_param_grads.append(torch_param.grad)

                assert lucid_param_grads, f"no params found for comparison {assert_has}"
                assert torch_param_grads, f"no params found for comparison {assert_has}"

                for g in lucid_param_grads:
                    assert np.isfinite(np.array(g)).all()
                for g in torch_param_grads:
                    assert np.isfinite(self._to_numpy(g)).all()

                lucid_optimizer.step()
                torch_optimizer.step()
                lucid_scheduler.step()
                torch_scheduler.step()

                if self.compare_scheduler:
                    self.assert_tensor_allclose(
                        [pg["lr"] for pg in lucid_optimizer.param_groups],
                        [pg["lr"] for pg in torch_optimizer.param_groups],
                        rtol=self.rtol,
                        atol=self.atol,
                    )

                for lucid_param, torch_param in self._iter_param_pairs(
                    lucid_model, torch_model
                ):
                    torch_grad = torch_param.grad
                    if torch_grad is not None:
                        torch_grad_arr = self._to_numpy(torch_grad)
                        if (
                            np.max(np.abs(torch_grad_arr))
                            < self.param_grad_skip_threshold
                        ):
                            continue

                    self.assert_tensor_allclose(
                        lucid_param,
                        torch_param.data,
                        rtol=self.param_rtol,
                        atol=self.param_atol,
                    )
