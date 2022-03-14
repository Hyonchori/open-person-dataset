# Extract frames by interval from images
import argparse
import os
import re
import glob
from pathlib import Path

import cv2
import numpy as np

SPLITS = {1: "train", 2: "test"}


def main(args):
    root = args.root
    save_dir = args.save_dir
    run_name = args.run_name
    target_split = args.target_split
    target_size = args.target_size
    save = args.save
    view = args.view
    frame_rate = args.frame_rate

    save_dir = increment_path(Path(save_dir) / run_name, exist_ok=False)
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)

    splits = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)) and x in SPLITS.values()]
    if target_split is not None:
        assert all(x in SPLITS for x in target_split), f"Some elements of {target_split} are not in {SPLITS}"
        splits = [SPLITS[x] for x in target_split]

    for split in splits:
        split_dir_path = os.path.join(root, split, split)
        cams_dict = {}
        for file_name in os.listdir(split_dir_path):
            if not file_name.endswith("jpg"):
                continue
            cam_name = "_".join(file_name.split("_")[:-1])
            if cam_name in cams_dict:
                cams_dict[cam_name].append(file_name)
            else:
                cams_dict[cam_name] = [file_name]
        for cam_name, imgs in cams_dict.items():
            extract_frames(split_dir_path, imgs, save_dir, target_size, save, frame_rate, view)


def extract_frames(img_dir_path, imgs, save_dir, target_size, save, frame_rate, view):
    imgs = sorted(imgs)
    if frame_rate is not None:
        target_frames = [int(len(imgs) * x) for x in frame_rate]
    for i, img_name in enumerate(imgs):
        if frame_rate is not None and i not in target_frames:
            continue
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        if target_size is not None and [w, h] != target_size:
            img = letterbox(img, target_size[::-1], auto=False)[0]
        if save:
            img_save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_save_path, img)
        if view:
            cv2.imshow("img", img)
            cv2.waitKey(0)


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/TVMP"
    parser.add_argument("--root", type=str, default=root)

    save_dir = "/media/daton/Data/datasets/TVMP/extracted_frames"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    run_name = "exp"
    parser.add_argument("--run-name", type=str, default=run_name)

    # 1: train, 2: test
    target_split = [1]
    target_split = None
    parser.add_argument("--target-split", type=int, default=target_split)

    target_size = [1280, 720]
    #target_size = None
    parser.add_argument("--target-size", type=int, default=target_size)

    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--view", action="store_true", default=False)
    parser.add_argument("--frame-rate", type=float, default=[0.5])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
