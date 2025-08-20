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

    is_mismatch = False
    for idx, ((tk, tv), (lk, lv)) in enumerate(
        zip(new_torch_dict.items(), lucid_dict.items()), start=1
    ):
        if tv.shape != lv.shape:
            if verbose:
                print(
                    f"{f"[{idx}]":>6}",
                    f"{f"❌ ({tk}) <-> ({lk}):":<80}",
                    f"{f"{tv.shape} <-> {lv.shape}":<30}",
                )
            is_mismatch = True
            continue

        if verbose:
            print(
                f"{f"[{idx}]":>6}",
                f"{f"✅ ({tk}) <-> ({lk}):":<80}",
                f"{f"{tv.shape} <-> {lv.shape}":<30}",
            )

    if is_mismatch:
        raise RuntimeError()

    for (_, tv), (lk, _) in zip(new_torch_dict.items(), lucid_dict.items()):
        lucid_dict[lk] = tv


def manual_convert(torch_dict, lucid_dict, torch_key_arr, lucid_key_arr, verbose=False):
    assert all(tk in torch_dict for tk in torch_key_arr)
    assert all(lk in lucid_dict for lk in lucid_key_arr)

    for tk, lk in zip(torch_key_arr, lucid_key_arr):
        torch_np = torch_dict[tk].numpy()
        if torch_np.shape != lucid_dict[lk].shape:
            raise ValueError(f"Shape mismatch: ({tk}) <-> ({lk})")

        lucid_dict[lk] = torch_np
        if verbose:
            print("[Manual]", f"{f"✅ ({tk}) <-> ({lk})":<80}")


import torch
import torchvision.models as models

torch_model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
torch_model.eval()
torch_dict = torch_model.state_dict()


import lucid
import lucid.models as models

lucid.random.seed(42)


lucid_model = models.densenet_201()
lucid_dict = lucid_model.state_dict()

convert_torch_to_lucid(
    torch_dict,
    lucid_dict,
    ignore=["num_batches_tracked", "transition", "norm5", "classifier"],
    verbose=True,
)

manual_convert(
    torch_dict,
    lucid_dict,
    torch_key_arr=[
        "features.transition1.norm.weight",
        "features.transition1.norm.bias",
        "features.transition1.conv.weight",
        "features.transition2.norm.weight",
        "features.transition2.norm.bias",
        "features.transition2.conv.weight",
        "features.transition3.norm.weight",
        "features.transition3.norm.bias",
        "features.transition3.conv.weight",
        "features.norm5.weight",
        "features.norm5.bias",
        "classifier.weight",
        "classifier.bias",
    ],
    lucid_key_arr=[
        "transitions.0.bn.weight",
        "transitions.0.bn.bias",
        "transitions.0.conv.weight",
        "transitions.1.bn.weight",
        "transitions.1.bn.bias",
        "transitions.1.conv.weight",
        "transitions.2.bn.weight",
        "transitions.2.bn.bias",
        "transitions.2.conv.weight",
        "bn_final.weight",
        "bn_final.bias",
        "fc.weight",
        "fc.bias",
    ],
    verbose=True,
)

lucid.save(lucid_dict, f"out/{lucid_model._alt_name}_in1k", safetensors=True)
