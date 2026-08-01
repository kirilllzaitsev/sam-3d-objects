import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sam3
import torch
from tqdm.auto import tqdm

from event_sam3d.config import (
    CO3D_DIR,
    CO3D_OBJECTS,
    MVSEC_DIR,
    MVSEC_SCENES,
    REPLICA_DIR,
    REPLICA_SCENES,
    SAM3_DIR,
    OBJ_DIR,
    OBJ_OBJECTS,
)

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
sam3_root = SAM3_DIR


import argparse

from event_sam3d.config import DATA_DIR
from event_sam3d.utils.misc_utils import get_ordered_paths
from event_sam3d.utils.sam3d_utils import save_sam3d_sparse_pred


def get_ds(ds_name, obj_name, filename=None, split=None, subsplit=None):
    from event_sam3d.datasets.ereplica_ds import EventReplicaDataset
    from event_sam3d.datasets.mvsec_ds import MVSECDataset
    from event_sam3d.datasets.obj_ds import ObjDataset
    from event_sam3d.datasets.co3d_ds import CO3DDataset
    from event_sam3d.datasets.rgbe_ds import RGBEDataset

    if ds_name == "mvsec":
        ds_cls = MVSECDataset
    elif ds_name == "ereplica":
        ds_cls = EventReplicaDataset
    elif ds_name == "co3d":
        ds_cls = CO3DDataset
    elif ds_name == "obj":
        ds_cls = ObjDataset
    else:
        ds_cls = RGBEDataset

    common_kwargs = dict(
        obj_name=obj_name,
        use_masks=True,
        use_vg_event_repr=True,
        len_limit=None,
        include_only_if_enough_events=False,
        min_num_events=1000,
    )
    if ds_name in ["mvsec", "ereplica", "co3d", "obj"]:
        other_kwargs = dict(
            seq_name=filename,
        )
    else:
        assert split is not None
        other_kwargs = dict(
            split=split,
            test_subsplit=subsplit,
            dirnames=None if filename is None else [filename],
        )
    dataset = ds_cls(
        transform=None,
        **other_kwargs,
        **common_kwargs,
    )
    return dataset


parser = argparse.ArgumentParser()
parser.add_argument("--ds_name", required=True, choices=["mvsec", "rgbe", "ereplica", "co3d", "obj"])
parser.add_argument("--part1", type=int, default=0)
parser.add_argument("--part2", type=int, default=0)
parser.add_argument("--do_debug", action='store_true')
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
is_co3d = ds_name == "co3d"
is_obj = ds_name == "obj"

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
elif is_co3d:
    meta = json.load(open(f"{CO3D_DIR}/meta.json"))
    N = 6
    obj_names = np.array_split(CO3D_OBJECTS, N)[args.part1]
    filenames_obj = defaultdict(list)
    for obj in obj_names:
        for inst in meta[obj]:
            filenames_obj[obj].append(f"{obj}/{inst}")
elif is_obj:
    meta = json.load(open(f"{OBJ_DIR}/meta.json"))
    N = 14
    obj_names = np.array_split(OBJ_OBJECTS, N)[args.part1]
    filenames_obj = defaultdict(list)
    for obj in obj_names:
        for inst in meta[obj][:4]:
            filenames_obj[obj].append(f"{obj}/{inst}")
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

for obj_name in tqdm(obj_names, desc=f"objects"):
    if is_co3d or is_obj:
        filenames=filenames_obj[obj_name]
    for filename in tqdm(filenames, desc=f"filenames for {obj_name}"):
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
            if ds_name == "mvsec":
                ds_dir=MVSEC_DIR
            elif is_rgbe:
                ds_dir=f"{DATA_DIR}/eventsam/RGBE-SEG/{args.stage}"
            elif is_ereplica:
                ds_dir=REPLICA_DIR
            elif is_co3d:
                ds_dir=CO3D_DIR
            else:
                ds_dir=OBJ_DIR
            save_dir = f"{ds_dir}/{filename}/sam3d_sparse"
            os.makedirs(save_dir, exist_ok=True)
            frame_name = sample["frame_name"]
            if is_co3d:
                save_path = f"{save_dir}/{frame_name}.pt"
            else:
                save_path = f"{save_dir}/{obj_name}_{frame_name}.pt"
            # print(f"{save_path}")
            if os.path.exists(save_path):
                continue
            with torch.no_grad():
                try:
                    output = inference._pipeline(
                    sample["rgb"], (sample["mask"] * 255).astype(np.uint8), seed=42,
                    use_stage1_distillation=True,
                    stage1_inference_steps=4,
                    use_stage2_distillation=True,
                    stage2_inference_steps=4
                )
                except Exception as e:
                    # if 'abc' in e, print and continue, else raise
                    if "Bounding" in str(e):
                        print(f"Error in inference for {filename}-{obj_name}-{frame_name}: {e}")
                        continue
                    else:
                        raise e

            save_sam3d_sparse_pred(save_path, output)

            if args.do_debug:
                print(save_path)
                exit(0)