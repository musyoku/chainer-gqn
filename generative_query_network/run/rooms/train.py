import argparse
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
import gqn


def main():
    dataset = gqn.data.Dataset(args.dataset_path)
    sampler = gqn.data.Sampler(dataset)
    iterator = gqn.data.Iterator(sampler, batch_size=32)

    for indices in iterator:
        images, viewpoints = dataset[indices]
        print(viewpoints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="rooms_dataset")
    args = parser.parse_args()
    main()
