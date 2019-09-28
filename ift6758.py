import argparse
import sys
from subprocess import call

sys.path.append('src/')

MODEL_PATH = "f7926f03b5b44121b9f3f106380f06b9"


def main(arguments: argparse.Namespace):
    call([
        "python", "src/predict.py",
        "-i", arguments.input_path,
        "-o", arguments.output_path,
        "-m", MODEL_PATH
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str)
    parser.add_argument("-o", "--output_path", type=str)
    args = parser.parse_args()
    main(args)
