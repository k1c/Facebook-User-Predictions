import argparse


def main(arguments: argparse.Namespace):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--age_estimator", type=str)
    parser.add_argument("--gender_estimator", type=str)
    parser.add_argument("--personality_estimator", type=str)
    args = parser.parse_args()
    main(args)
