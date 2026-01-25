import argparse
import copy
import datetime as dt
import functools
import itertools
import json
import logging
import math
import os
import pickle
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sam3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from event_sam3d.config import MVSEC_DIR, MVSEC_SCENES, REPLICA_DIR, REPLICA_SCENES, RGBE_DIR, SAM3_DIR
from event_sam3d.nb_utils_static import get_ds

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
sam3_root = SAM3_DIR


import argparse

from event_sam3d.config import DATA_DIR
from event_sam3d.datasets.mvsec_ds import MVSECDataset
from event_sam3d.nb_utils_static import fetch_samples
from event_sam3d.utils.misc_utils import get_ordered_paths
from event_sam3d.utils.sam3d_utils import save_sam3d_sparse_pred
from event_sam3d.utils.segm_utils import get_sam3_model, get_sam3_preds, save_sam3_pred
from event_sam3d.utils.vis_utils import make_grid_image

parser = argparse.ArgumentParser()
parser.add_argument("--ds_name", required=True, choices=["mvsec", "rgbe", "ereplica"])
parser.add_argument("--part1", type=int, default=0)
parser.add_argument("--part2", type=int, default=0)
parser.add_argument(
    "--stage", required=False, choices=["train", "test-normal"], default="train"
)
parser.add_argument(
    "--test_subsplit", required=False, choices=["easy", "medium", "hard"], default="easy"
)
args, _ = parser.parse_known_args()
print(f"{args=}")


ds_name = args.ds_name
is_mvsec = ds_name == "mvsec"
is_rgbe = ds_name == "rgbe"
is_ereplica = ds_name == "ereplica"

if is_rgbe:
    stage = args.stage
    dirs = get_ordered_paths(f"{DATA_DIR}/eventsam/RGBE-SEG/{stage}/*")
    prompts = [
        "person",
        "car",
        # "object",
        # "animal",
        # "device",
        # "drone",
        # infrequent
        # "hydrant",
        # "lamp",
        # "book",
        # "tv",
    ]
    assert args.part1 in [0, 1, 2, 3]
    assert args.part2 in [0, 1]
    obj_names = prompts[args.part2 : args.part2 + 1]
    dirs = [Path(x).name for x in get_ordered_paths(f"{DATA_DIR}/eventsam/RGBE-SEG/{stage}/*") if Path(x).is_dir()]
    dirs = [x for x in dirs if x not in ['sam3d']]
    filenames = np.array_split(dirs, 4)[args.part1]
    print(f"Running part {args.part1}/{4}, {len(filenames)} dirs")
elif is_mvsec:
    assert args.part1 in [0, 1, 2, 3]
    assert args.part2 in [0, 1]
    obj_names = ["barrel", "rug"][args.part2 : args.part2 + 1]
    filenames=MVSEC_SCENES[args.part1 : args.part1 + 1]
else:
    prompts1 = [
        "chair",
        "table",
        "sofa",
        "pillow",
        "door",
    ]
    prompts2 = [
        "cabinet",
        "lamp",
        "plant",
        "vase",
        "mirror",
        "whiteboard",
        "painting",
    ]
    filenames=REPLICA_SCENES[args.part1 : args.part1 + 1]
    obj_names = prompts1 if args.part2 == 0 else prompts2

from event_sam3d.config import SAM3D_DIR

sys.path.append(f"{SAM3D_DIR}/notebook")
from inference import Inference

config_path = f"{SAM3D_DIR}/checkpoints/hf/pipeline.yaml"
device = "cpu"
device = "cuda"
inference = Inference(
    config_path,
    compile=False,
    use_event=False,
    # rgbe_fusion_type=args.rgbe_fusion_type,
    device=device,
    use_ckpt=True,
    use_only_sparse=True,
    # ss_generator_cond_embedder_ckpt_path=f"{ckpt_dir}/best_ss_generator_cond_embedder.pt",
    # rgbe_fuser_ckpt_path=f"{ckpt_dir}/best_rgbe_fuser.pt",
)
inference._pipeline.eval()
torch.inference_mode().__enter__()

for filename in filenames:
    for obj_name in tqdm(obj_names, desc=f"Processing {filename}"):
        ds = get_ds(
            ds_name=ds_name,
            obj_name=obj_name,
            filename=filename,
            split=args.stage,
            subsplit=args.test_subsplit,
        )
        if len(ds) == 0:
            continue
        for idx in (
            tqdm(
                range(len(ds)),
                desc=f"Running inference on {filename}-{obj_name}",
                total=len(ds),
            )
        ):
            sample = ds[idx]
            ds_dir=MVSEC_DIR if ds_name == "mvsec" else (REPLICA_DIR if is_ereplica else f"{DATA_DIR}/eventsam/RGBE-SEG/{args.stage}")
            save_dir = f"{ds_dir}/{filename}/sam3d_sparse"
            os.makedirs(save_dir, exist_ok=True)
            frame_name = sample["frame_name"]
            save_path = f"{save_dir}/{obj_name}_{frame_name}.pt"
            if os.path.exists(save_path):
                continue
            with torch.no_grad():
                output = inference._pipeline(
                    sample["rgb"], (sample["mask"] * 255).astype(np.uint8), seed=42,
                    use_stage1_distillation=True,
                    stage1_inference_steps=4,
                    use_stage2_distillation=True,
                    stage2_inference_steps=4
                )

            save_sam3d_sparse_pred(save_path, output)

    #         print(save_path)
    #         break
    #     break
    # break