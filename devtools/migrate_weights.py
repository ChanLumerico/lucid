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
        raise ValueError()

    for (_, tv), (lk, _) in zip(new_torch_dict.items(), lucid_dict.items()):
        lucid_dict[lk] = tv


import torch
import torchvision.models as models

torch_model = models.resnext101_64x4d(
    weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
)
torch_model.eval()
torch_dict = torch_model.state_dict()


import lucid
import lucid.models as models

lucid.random.seed(42)


lucid_model = models.resnext_101_64x4d()
lucid_dict = lucid_model.state_dict()

convert_torch_to_lucid(
    torch_dict,
    lucid_dict,
    ignore=["num_batches_tracked"],
    verbose=True,
)

lucid.save(lucid_dict, f"out/{lucid_model._alt_name}_in1k", safetensors=True)
