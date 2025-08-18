import argparse

from lucid import weights as W


p = argparse.ArgumentParser()

p.add_argument("model_key")
p.add_argument("--tag", default="DEFAULT")

args = p.parse_args()

en = W.get_enum(args.model_key)
entry = getattr(en, args.tag).value
state = W.load_state_dict_from_url(args.model_key, entry)
keys = list(state.keys())

num = sum(getattr(v, "size", 1) if hasattr(v, "shape") else 1 for v in state.values())

print("ok", args.model_key, args.tag, len(keys), num)
