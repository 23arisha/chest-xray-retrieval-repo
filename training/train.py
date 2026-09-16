import argparse
import datetime
import json
import os
import sys

import dateutil.tz
import pandas as pd
import torch
from transformers import BertTokenizer

from configs.train_config import cfg
from src.data.datasets import build_dataset_itm, build_dataset_mlm, collate_fn_ignore_none
from src.data.preprocessing import build_sentence_dict_from_rows, deduplicate_sentences
from src.data.samplers import UniqueImageSampler
from src.engine.trainer import JoImTeR
from src.utils import mkdir_p

# Default Kaggle paths used for the reported run. Override via CLI args for
# a local setup (see configs/datasets.md for dataset sources).
DEFAULT_MIMIC_CSV = "/kaggle/input/datasets/shery432/mimic-file/mimic-data.csv"
DEFAULT_BAD_IMAGE_PATH = (
    "/kaggle/input/datasets/zainabhalhoul/mimic-dataset/"
    "mimicDatatotal/00771/00771.png"
)
DEFAULT_OUTPUT_ROOT = "/kaggle/working/output"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic_csv", default=DEFAULT_MIMIC_CSV,
                         help="Path to the MIMIC-CXR findings-report CSV "
                              "(columns: image_path, findings).")
    parser.add_argument("--bad_image_path", default=DEFAULT_BAD_IMAGE_PATH,
                         help="A known-corrupt/missing image path to exclude, if any.")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT,
                         help="Root directory under which a timestamped run "
                              "directory (checkpoints, logs, tensorboard) is created.")
    parser.add_argument("--train", action="store_true",
                         help="Actually launch training. Without this flag, the "
                              "script builds the datasets/dataloaders/trainer and "
                              "exits, for inspection or debugging.")
    return parser.parse_args()


def build_output_dir(output_root):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = f"{output_root}/{cfg.DATASET_NAME}_{cfg.CONFIG_NAME}_{timestamp}"
    mkdir_p(output_dir)

    cfg_log = os.path.join(output_dir, "cfg.log")
    with open(cfg_log, "w") as f:
        json.dump(cfg.__dict__, f, indent=4)

    return output_dir


def load_mimic(mimic_csv, bad_image_path):
    mimic = pd.read_csv(mimic_csv)
    if bad_image_path:
        mimic = mimic[mimic["image_path"] != bad_image_path].reset_index(drop=True)
    print(len(mimic))
    return mimic


def main():
    args = parse_args()

    torch.manual_seed(cfg.seed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(cfg.seed)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    mimic = load_mimic(args.mimic_csv, args.bad_image_path)
    output_dir = build_output_dir(args.output_root)

    kept_rows, df_clean = deduplicate_sentences(mimic, max_per_sentence=3)
    dataset = build_sentence_dict_from_rows(
        kept_rows, df_clean, tokenizer, max_length=cfg.max_length)

    print(f"train keys: {len(dataset['data_split']['train_uids'])}")
    print(f"val keys:   {len(dataset['data_split']['val_uids'])}")
    print(f"test keys:  {len(dataset['data_split']['test_uids'])}")
    assert len(dataset['data_split']['train_uids']) > 0, "train split is empty!"
    assert len(dataset['data_split']['val_uids']) > 0, "val split is empty!"

    total_bytes = sum(
        sys.getsizeof(v['image_path']) for v in dataset['data_dict'].values()
    )
    print(f'image_path RAM: {total_bytes / 1e6:.2f} MB')

    data_set_itm = build_dataset_itm(dataset, 'train', cfg, output_dir)
    itm_sampler = UniqueImageSampler(data_set_itm)

    train_loader_itm = torch.utils.data.DataLoader(
        data_set_itm,
        batch_size=cfg.batch_size,
        sampler=itm_sampler,
        shuffle=False,
        collate_fn=collate_fn_ignore_none,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    data_set_mlm = build_dataset_mlm(dataset, 'train', cfg)
    train_loader_mlm = torch.utils.data.DataLoader(
        data_set_mlm,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn_ignore_none,
        drop_last=True,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_data_set_itm = build_dataset_itm(dataset, 'val', cfg, output_dir)
    val_loader_itm = torch.utils.data.DataLoader(
        val_data_set_itm,
        batch_size=cfg.val_batch_size,
        collate_fn=collate_fn_ignore_none,
        drop_last=False,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_data_set_mlm = build_dataset_mlm(dataset, 'val', cfg)
    val_loader_mlm = torch.utils.data.DataLoader(
        val_data_set_mlm,
        batch_size=cfg.val_batch_size,
        collate_fn=collate_fn_ignore_none,
        drop_last=False,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f'Vocab size: {data_set_itm.vocab_size}')
    print(f'ITM train:  {len(data_set_itm)}')
    print(f'MLM train:  {len(data_set_mlm)}')
    print(f'ITM val:    {len(val_data_set_itm)}')
    print(f'MLM val:    {len(val_data_set_mlm)}')

    algo = JoImTeR(output_dir, train_loader_itm, train_loader_mlm,
                   val_loader_itm, val_loader_mlm)

    if args.train:
        import time
        start_t = time.time()
        algo.train()
        print('Total training time: %.1f sec' % (time.time() - start_t))
    else:
        print("Dataset/dataloaders/trainer built. Pass --train to launch training.")


if __name__ == "__main__":
    main()
