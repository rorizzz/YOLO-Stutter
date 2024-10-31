import argparse
import warnings
from trainer import Trainer 

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="Speech Text alignments")

    # model arguments
    parser.add_argument("--in-channels", "-ic", type=int, default=1)
    parser.add_argument("--out-channels", "-oc", type=int, default=256)
    parser.add_argument("--kernel-size", "-ks", type=int, default=3)
    parser.add_argument("--kernel-stride", "-kst", type=int, default=1)
    parser.add_argument("--num-decoder-blocks", "-d", type=int, default=4)
    parser.add_argument("--num-classes", "-nc", type=int, default=4)
    parser.add_argument("--downsample-factor", "-df", type=int, default=16)

    # training/config arguments
    parser.add_argument(
        "--config-path", "-cp", type=str, default="./utils/vits/configs/ljs_base.json"
    )
    parser.add_argument("--epochs", "-e", type=int, default=50)
    parser.add_argument("--lr", "-lr", type=float, default=3e-4)
    parser.add_argument("--num-steps", "-ns", type=int, default=10)
    parser.add_argument("--batch-size", "-bs", type=int, default=64)

    args = parser.parse_args()

    t = Trainer()
    t.train_and_eval()
