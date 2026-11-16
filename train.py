import argparse
import datetime
import json
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.nersemble import NeRSembleData
from joint_trainer import JointTrainer
from hair_trainer import HairTrainer
from utils import seed_everything
from exp.config_utils import effective_fresco_config, normalize_dotted_config
from exp.cross_view_sampler import SameFramePairBatchSampler
from exp.fresco_stage_schedule import apply_subject460_stage, paired_sampler_required, schedule_enabled

from exp.lfimg_loss import compute_lfimg_loss
from exp.lfuv_loss import compute_lfuv_loss
from exp.uv_seam_loss import compute_seam_loss
from exp.masks import get_face_vis_mask
from exp.uv_bake import bake_to_uv_functional, get_uv_resolution, make_face_px_mask

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--config_path", default="./params.yaml", type=str)
parser.add_argument("--workspace", type=str, required=True)
parser.add_argument("--version", type=str, required=True)
parser.add_argument("--extra_config", type=str, default="{}", required=False)
parser.add_argument("--time", action="store_true")
args = parser.parse_args()

with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

extra_config = json.loads(args.extra_config)
for k in extra_config.keys():
    if '.' in k:
        parent_key = k.split('.')[0]
        if parent_key not in config:
            config[parent_key] = {}
    else:
        assert k in config, k
config.update(extra_config)
config = normalize_dotted_config(config, extra_config)
if schedule_enabled(config):
    # Start with the epoch-0 resource state. A resumed trainer reapplies the
    # stage selected by the checkpoint's next_epoch before its first batch.
    apply_subject460_stage(config, 0)
config["bald"] = args.config_path.split("/")[-2] in ["218"]

tmp_config_path = os.path.join(os.path.dirname(args.config_path), "params_tmp.yaml")
with open(tmp_config_path, "w") as f:
    print("Dumping extra config file...")
    yaml.dump(config, f)

config["current_epoch"] = 0

gpu_id = int(config.get("training.gpus", "0"))
torch.cuda.set_device(gpu_id)


def _make_train_loader(train_set, logger, paired):
    batch_size = config["data.per_gpu_batch_size"]
    if paired:
        pair_sampler = SameFramePairBatchSampler(train_set, batch_size, shuffle=True, drop_last=False)
        loader = DataLoader(train_set, batch_sampler=pair_sampler, num_workers=config["data.num_workers"])
        if logger:
            logger.info(
                "Cross-view sampler: dataset_images=%d eligible_images=%d emitted_images=%d "
                "sample_delta=%+d excluded_single_view=%d batches_per_epoch=%d",
                pair_sampler.num_dataset_samples, pair_sampler.num_source_samples,
                pair_sampler.num_emitted_samples, pair_sampler.epoch_sample_delta,
                pair_sampler.num_excluded_single_view_samples, len(pair_sampler),
            )
        return loader
    return DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=config["data.num_workers"], drop_last=False,
    )


class DynamicTrainLoader:
    """Rebuild the loader only when the curriculum changes sampler type."""

    def __init__(self, train_set, logger, initial_loader):
        self.train_set = train_set
        self.logger = logger
        self.loader = initial_loader
        self.paired = isinstance(getattr(initial_loader, "batch_sampler", None), SameFramePairBatchSampler)

    def __call__(self, epoch, current_config):
        paired = paired_sampler_required(current_config)
        if paired != self.paired:
            self.loader = _make_train_loader(self.train_set, self.logger, paired)
            self.paired = paired
            self.logger.info(
                "Epoch %d switched training sampler to %s",
                epoch, type(self.loader.batch_sampler).__name__ if paired else "RandomSampler",
            )
        return self.loader


def get_dataset(logger, datatype="nersemble"):
    data_dict = {"nersemble": NeRSembleData}
    assert datatype in data_dict.keys(), "Not Supported Datatype: {}".format(datatype)
    Data = data_dict[datatype]

    # get dataset
    batch_size = config["data.per_gpu_batch_size"]
    train_set = Data(config)
    if logger:
        logger.info("number of train images: {}".format(len(train_set)))
    train_loader = _make_train_loader(train_set, logger, paired_sampler_required(config))

    val_set = Data(config, split="val")
    if logger:
        logger.info("number of val images: {}".format(len(val_set)))
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data.num_workers"],
        drop_last=True,
    )

    return train_loader, val_loader, train_set.radius, train_set.load_all_flame_params()


def train():
    seed_everything(42)

    # Enable cudnn benchmark for speed optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Config logging and tb writer
    logger = None
    import logging

    # logging to file and stdout
    config["log_file"] = os.path.join(args.workspace, args.version, "training.log")
    logger = logging.getLogger("MeGA")
    file_handler = logging.FileHandler(config["log_file"])
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.handlers = [file_handler, stream_handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info("Training config: {}".format(config))
    logger.info("Effective FReSCO config: {}".format(effective_fresco_config(config)))

    # tensorboard summary_writer
    config["tb_writer"] = SummaryWriter(log_dir=config["local_workspace"])
    config["logger"] = logger

    # Init data loader
    config["datatype"] = args.config_path.split("/")[-3]
    train_loader, val_loader, radius, all_flame_params = get_dataset(logger, config["datatype"])

    if config.get('exp.lf_img.enable', False):
        logger.info("J1 LF-IMG loss enabled")
    if config.get('exp.lf_uv.enable', False):
        logger.info("J2 LF-UV loss enabled")

    Trainer = HairTrainer if config["pipe.neutral_hair"] else JointTrainer
    print("[DEBUG] train.py: Creating trainer")
    logger.info("[DEBUG] train.py: Creating trainer")
    trainer = Trainer(config, logger, radius, all_flame_params=all_flame_params)
    print("[DEBUG] train.py: Trainer created, about to call trainer.train()")
    logger.info("[DEBUG] train.py: Trainer created, about to call trainer.train()")
    train_loader_provider = None
    if schedule_enabled(config):
        train_loader_provider = DynamicTrainLoader(train_loader.dataset, logger, train_loader)
    if isinstance(trainer, JointTrainer):
        trainer.train(
            train_loader, val_loader, show_time=args.time,
            train_loader_provider=train_loader_provider,
        )
    else:
        if train_loader_provider is not None:
            raise RuntimeError("The Subject 460 FReSCO schedule requires JointTrainer")
        trainer.train(train_loader, val_loader, show_time=args.time)


def main():
    workspace = os.path.join(args.workspace, args.version)
    config["local_workspace"] = workspace
    if not os.path.exists(workspace):
        os.makedirs(workspace)
    shutil.copy(tmp_config_path, os.path.join(workspace, "params.yaml"))

    train()


if __name__ == "__main__":
    main()
