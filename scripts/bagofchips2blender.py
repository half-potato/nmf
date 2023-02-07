import torch
from icecream import ic
import argparse
from pathlib import Path
import json

def create_matrix(vals):
    T = torch.eye(4)
    T[0, 3] = vals[0]
    T[1, 3] = vals[1]
    T[2, 3] = vals[2]
    T[:3, :3] = torch.tensor(vals[3:]).reshape(3, 3)
    return T

def start_dictionary():
    # some extra stuff
    focal_length = 541.961
    meta = dict(
        cx = 320,
        cy = 240,
        h = 480,
        w = 640,
        white_bg = False,
        near_far = [],
        fl_x = focal_length,
        fl_y = focal_length,
    )
    return meta

def create_meta(path, skip_n=10):
    train_frames = []
    test_frames = []
    with open(path / 'xy_poses.txt', 'r') as f:
        N = f.readline()
        for l in f.readlines():
            vals = [float(x) for x in l.split(' ')]
            n = int(vals[0])
            mat = create_matrix(vals[1:])
            frame = dict(
                file_path=f"./rgb/{n:04d}",
                transform_matrix=mat.tolist(),
            )
            if n % skip_n == 1:
                train_frames.append(frame)
            if n % skip_n == 5:
                test_frames.append(frame)
    train_meta = start_dictionary()
    train_meta['frames'] = train_frames
    test_meta = start_dictionary()
    test_meta['frames'] = test_frames
    return train_meta, test_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    train_meta, test_meta = create_meta(args.path)
    with (args.path / "transforms_train.json").open('w') as f:
        json.dump(train_meta, f, indent=4)

    with (args.path / "transforms_test.json").open('w') as f:
        json.dump(test_meta, f, indent=4)
