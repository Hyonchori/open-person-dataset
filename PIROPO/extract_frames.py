# Extract frames by selection (input Keyboard 's')
import argparse
import os
import re
import glob
from pathlib import Path

import cv2
import numpy as np

SPLITS = {1: "omni_1A", 2: "omni_1B", 3: "omni_2A", 4: 'omni_3A'}


def main(args):
    root = args.root
    save_dir = args.save_dir
    run_name = args.run_name
    target_split = args.target_split
    target_size = args.target_size
    crop_rate = args.crop_rate
    save = args.save

    save_dir = increment_path(Path(save_dir) / run_name, exist_ok=False)
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)

    splits = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)) and x in SPLITS.values()]
    if target_split is not None:
        assert all(x in SPLITS for x in target_split), f"Some elements of {target_split} are not in {SPLITS}"
        splits = [SPLITS[x] for x in target_split]

    for split in splits:
        split_dir_path = os.path.join(root, split)
        scenes = [x for x in os.listdir(split_dir_path) if os.path.isdir(os.path.join(split_dir_path, x)) and
                  "omni" in x]

        for scene in scenes:
            scene_dir_path = os.path.join(split_dir_path, scene)
            extract_selected_frames(scene_dir_path, scene, target_size, crop_rate, save_dir, save)


def extract_selected_frames(img_dir_path, vid_name, target_size, crop_rate, save_dir, save):
    print(f"\n--- Processing {vid_name}")
    imgs = [x for x in sorted(os.listdir(img_dir_path)) if x.endswith(".jpg")]
    cnt = 0
    tmp_idx = 0
    while True:
        if tmp_idx >= len(imgs):
            break
        img_name = imgs[tmp_idx]
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)

        h, w, _ = img.shape
        if crop_rate is not None:
            img = img[int(h * crop_rate[1][0]): int(h * crop_rate[1][1]),
                      int(w * crop_rate[0][0]): int(w * crop_rate[0][1])]
        if target_size is not None and [w, h] != target_size:
            img = letterbox(img, target_size[::-1], auto=False)[0]

        info = f"{vid_name}: {tmp_idx + 1} / {len(imgs)}"
        imv = img.copy()
        plot_info(imv, info)
        cv2.imshow("img", imv)
        tmp_input = cv2.waitKey(0) & 0xFF
        if tmp_input == ord('s'):
            save_path = os.path.join(save_dir, img_name)
            if save:
                cv2.imwrite(save_path, img)
            tmp_idx += 1
            cnt += 1
        elif tmp_input == ord('a'):  # a: (<-)
            tmp_idx = max(0, tmp_idx - 10)
        elif tmp_input == ord('d'):  # d: (->)
            tmp_idx = min(tmp_idx + 10, len(imgs))
        elif tmp_input == ord('q'):
            break
        else:
            tmp_idx += 1
    print(f"\t{cnt} images are saved!")


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


def plot_info(img, info, font_size=2, font_thickness=2):
    label_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] * 2), [0, 0, 0], -1)
    cv2.putText(img, info, (5, int(label_size[1] * 1.5))
                , cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)


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

    root = "/media/daton/Data/datasets/PIROPO"
    parser.add_argument("--root", type=str, default=root)

    save_dir = "/media/daton/Data/datasets/PIROPO/extracted_frames"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    run_name = "exp"
    parser.add_argument("--run-name", type=str, default=run_name)

    # 1: omni_1A, 2: omni_1B, 3: omni_2A, 4: omni_3A
    target_split = 1
    target_split = None
    parser.add_argument("--target-split", type=str, default=target_split)

    target_size = [1280, 720]
    parser.add_argument("--target-size", type=int, default=target_size)

    crop_rate = [(0.2, 0.8), (0.2, 0.8)]
    parser.add_argument("--crop-rate", type=float, default=crop_rate)

    parser.add_argument("--save", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
