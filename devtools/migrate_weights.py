def print_state_dict(state_dict):
    print()
    for k, v in state_dict.items():
        print(f"({k}): {v.shape}")


def convert_torch_to_lucid(torch_dict, lucid_dict, ignore=[], verbose=False):
    new_torch_dict = {}
    for tk, tv in torch_dict.items():
        if any(ig in tk for ig in ignore):
            continue
        new_torch_dict[tk] = tv.numpy()

    mismatch_keys = []
    for idx, ((tk, tv), (lk, lv)) in enumerate(
        zip(new_torch_dict.items(), lucid_dict.items()), start=1
    ):
        if tv.shape != lv.shape:
            mismatch_keys.append(lk)
            if verbose:
                print(
                    f"{f"[{idx}]":>6}",
                    f"{f"❌ ({tk}) <-> ({lk}):":<80}",
                    f"{f"{tv.shape} <-> {lv.shape}":<30}",
                )
            continue

        if verbose:
            print(
                f"{f"[{idx}]":>6}",
                f"{f"✅ ({tk}) <-> ({lk}):":<80}",
                f"{f"{tv.shape} <-> {lv.shape}":<30}",
            )

    for (_, tv), (lk, _) in zip(new_torch_dict.items(), lucid_dict.items()):
        if lk in mismatch_keys:
            continue
        lucid_dict[lk] = tv


def manual_convert(torch_dict, lucid_dict, torch_key_arr, lucid_key_arr, verbose=False):
    assert all(tk in torch_dict for tk in torch_key_arr)
    assert all(lk in lucid_dict for lk in lucid_key_arr)

    for tk, lk in zip(torch_key_arr, lucid_key_arr):
        torch_np = torch_dict[tk].numpy()
        if torch_np.shape != lucid_dict[lk].shape:
            try:
                torch_np.reshape(*lucid_dict[lk].shape)
            except Exception:
                raise ValueError(f"Shape mismatch: ({tk}) <-> ({lk})")

        lucid_dict[lk] = torch_np.reshape(*lucid_dict[lk].shape)
        if verbose:
            print("[Manual]", f"{f"✅ ({tk}) <-> ({lk}):":<80} {lucid_dict[lk].shape}")


import torch
import torchvision.models as models

torch_model = models.mobilenet_v3_large(
    weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
)
torch_model.eval()
torch_dict = torch_model.state_dict()


import lucid
import lucid.models as models

lucid.random.seed(42)


lucid_model = models.mobilenet_v3_large()
lucid_dict = lucid_model.state_dict()

convert_torch_to_lucid(
    torch_dict,
    lucid_dict,
    ignore=["num_batches_tracked"],
    verbose=True,
)

manual_convert(
    torch_dict,
    lucid_dict,
    torch_key_arr=[
        "features.11.block.2.fc1.weight",
        "features.11.block.2.fc2.weight",
        "features.12.block.2.fc1.weight",
        "features.12.block.2.fc2.weight",
        "features.13.block.2.fc1.weight",
        "features.13.block.2.fc2.weight",
        "features.14.block.2.fc1.weight",
        "features.14.block.2.fc2.weight",
        "features.15.block.2.fc1.weight",
        "features.15.block.2.fc2.weight",
    ],
    lucid_key_arr=[
        "bottlenecks.10.residual.2.fc1.weight",
        "bottlenecks.10.residual.2.fc2.weight",
        "bottlenecks.11.residual.2.fc1.weight",
        "bottlenecks.11.residual.2.fc2.weight",
        "bottlenecks.12.residual.2.fc1.weight",
        "bottlenecks.12.residual.2.fc2.weight",
        "bottlenecks.13.residual.2.fc1.weight",
        "bottlenecks.13.residual.2.fc2.weight",
        "bottlenecks.14.residual.2.fc1.weight",
        "bottlenecks.14.residual.2.fc2.weight",
    ],
    verbose=True,
)

lucid.save(lucid_dict, f"out/{lucid_model._alt_name}_in1k", safetensors=True)
