import argparse
import os

import torch
from inference import Inference

from event_sam3d.config import MVSEC_DIR
from event_sam3d.datasets.mvsec_ds import MVSEC
from event_sam3d.utils.common_utils import cast_to_numpy

parser = argparse.ArgumentParser()
parser.add_argument("--seqname", default="indoor_flying4_data")
parser.add_argument("--use_encoder_only", action="store_true")
args, _ = parser.parse_known_args()

PATH = os.getcwd()
TAG = "hf"
if args.use_encoder_only:
    config_path = f"{PATH}/../checkpoints/{TAG}/pipeline_encoder.yaml"
else:
    config_path = f"{PATH}/../checkpoints/{TAG}/pipeline.yaml"


name = args.seqname
ds = MVSEC(seq_name=name)
print(len(ds))
idx = 500
image = ds[idx]["rgb"]
state = torch.load(f"{MVSEC_DIR}/{name}/sam3/barrel_{idx:06d}.pt")
mask = cast_to_numpy(state["masks"][0].squeeze())

# display_image(image, masks=[mask])

inference = Inference(config_path, compile=False)
output = inference(image, mask, seed=42)

name = "tmp"
if args.use_encoder_only:
    ...
else:
    save_path = f"{PATH}/gaussians/single/{name}.ply"
    output["gs"].save_ply(save_path)
    print(f"saved to {save_path}")
