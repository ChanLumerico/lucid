def print_state_dict(state_dict):
    print()
    for k, v in state_dict.items():
        print(f"({k}): {v.shape}")


def match_two_dicts(torch_dict, lucid_dict):
    for (tk, tv), (lk, lv) in zip(torch_dict.items(), lucid_dict.items()):
        tvn = tv.numpy()
        if tvn.shape != lv.shape:
            print(f"Shape mismatch at: ({tk}) <-> ({lk}) – {tvn.shape} != {lv.shape}")
            break


def convert_torch_to_lucid(torch_dict, lucid_dict):
    for (_, tv), (lk, _) in zip(torch_dict.items(), lucid_dict.items()):
        lucid_dict[lk] = tv.numpy()


import torch
import torchvision.models as models

torch_model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
torch_model.eval()
torch_dict = torch_model.state_dict()


import lucid
import lucid.models as models


lucid_model = models.vggnet_13()
lucid_dict = lucid_model.state_dict()


match_two_dicts(torch_dict, lucid_dict)

convert_torch_to_lucid(torch_dict, lucid_dict)

lucid.save(lucid_dict, "out/vggnet_13_in1k", safetensors=True)
