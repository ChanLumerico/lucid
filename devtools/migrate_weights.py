def print_state_dict(state_dict):
    print()
    for k, v in state_dict.items():
        print(f"({k}): {v.shape}")


def match_two_dicts(torch_dict, lucid_dict, ignore=[]):
    for (tk, tv), (lk, lv) in zip(torch_dict.items(), lucid_dict.items()):
        if any(term in tk for term in ignore):
            continue

        tvn = tv.numpy()
        if tvn.shape != lv.shape:
            print(
                f"{f"❌ ({tk}) <-> ({lk}):":<80}",
                f"{f"{tvn.shape} <-> {lv.shape}":<30}",
            )
            continue
        print(
            f"{f"✅ ({tk}) <-> ({lk}):":<80}",
            f"{f"{tvn.shape} <-> {lv.shape}":<30}",
        )


def convert_torch_to_lucid(torch_dict, lucid_dict):
    for (_, tv), (lk, _) in zip(torch_dict.items(), lucid_dict.items()):
        lucid_dict[lk] = tv.numpy()


import torch
import torchvision.models as models

torch_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
torch_model.eval()
torch_dict = torch_model.state_dict()


import lucid
import lucid.models as models


lucid_model = models.resnet_18()
lucid_dict = lucid_model.state_dict()


match_two_dicts(torch_dict, lucid_dict, ignore=["num_batches_tracked"])

convert_torch_to_lucid(torch_dict, lucid_dict)

lucid.save(lucid_dict, "out/vggnet_19_in1k", safetensors=True)
