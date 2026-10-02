"""Train a PCPH-BigVGAN vocoder.

python train.py --name my_vocoder --filelist data/filelist.txt --epochs 100
"""

import argparse
import json
import os

from pcph_bigvgan.data import CHECKPOINT_MODES, run_dir
from pcph_bigvgan.train import main


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True, help="Run name; everything is written to logs/<name>.")
    parser.add_argument("--filelist", required=True,
                        help="Rows of 'audio|f0.npy', or an RVC filelist.txt (audio|features|f0|f0_voiced|speaker).")
    parser.add_argument("--data-root", default=None,
                        help="Folder the filelist's relative paths start from. Default: the working directory.")
    parser.add_argument("--config", default=None,
                        help="Config copied to logs/<name>/config.json on the first run. Default: configs/44100.json.")
    parser.add_argument("--epochs", type=int, required=True, help="Total epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--save-every", type=int, default=1, help="Save every N epochs.")
    parser.add_argument("--gpu", default="0", help="GPU ids joined by '-', e.g. 0-1.")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--checkpoints", choices=CHECKPOINT_MODES, default="latest",
                        help="Training checkpoints kept; exports are always kept.")
    parser.add_argument("--fresh", action="store_true", help="Ignore the run's checkpoints instead of resuming.")
    parser.add_argument("--pretrained-g", default="", help="Generator checkpoint or export to start from.")
    parser.add_argument("--pretrained-d", default="", help="Discriminator checkpoint to start from.")
    parser.add_argument("--mel-degradation", type=float, default=0.0,
                        help="0 to 1: train the generator on mels degraded like a generated one.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    spec = {
        "model_name": args.name,
        "filelist": os.path.abspath(args.filelist),
        "data_root": os.path.abspath(args.data_root) if args.data_root else os.getcwd(),
        "config": os.path.abspath(args.config) if args.config else None,
        "total_epochs": args.epochs,
        "batch_size": args.batch_size,
        "save_every": args.save_every,
        "gpu": args.gpu,
        "precision": args.precision,
        "checkpoints": args.checkpoints,
        "fresh": args.fresh,
        "pretrained_g": args.pretrained_g,
        "pretrained_d": args.pretrained_d,
        "mel_degradation": args.mel_degradation,
    }
    os.makedirs(run_dir(args.name), exist_ok=True)
    spec_path = os.path.join(run_dir(args.name), "spec.json")
    with open(spec_path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=4)
    main(spec_path)
