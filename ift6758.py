import argparse
import sys
from subprocess import call

sys.path.append('src/')

from util.utils import get_current_timestamp

MODEL_PATH = 'models/cc1e2f23c4a44f0d87c6840192360c63_2019-11-11_02.28.58.pkl'


def main(arguments: argparse.Namespace):
    call([
        "python", "src/predict.py",
        "-i", arguments.input_path,
        "-o", arguments.output_path,
        "-m", MODEL_PATH
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_output_path = "predictions/{}/".format(get_current_timestamp())
    parser.add_argument("-i", "--input_path", type=str, default="../new_data/Public_Test/")
    parser.add_argument("-o", "--output_path", type=str, default=default_output_path)
    args = parser.parse_args()
    main(args)
