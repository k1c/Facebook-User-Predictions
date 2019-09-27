import argparse
import sys
from subprocess import call

sys.path.append('src/')

MODEL_PATH = "f7926f03b5b44121b9f3f106380f06b9"


def main(arguments: argparse.Namespace):
    call([
        "python", "src/predict.py",
        "-i", arguments.data_path,
        "-o", arguments.save_path,
        "-m", MODEL_PATH
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_path", type=str)
    parser.add_argument("-o", "--save_path", type=str)
    args = parser.parse_args()
    main(args)
