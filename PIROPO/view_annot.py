# View annotations of 'People in Indoor Rooms with Perspective and Omnidirectional cameras'
# we focus on 'omni view'
# from https://sites.google.com/site/piropodatabase/
import argparse
import os

import cv2
import numpy as np

SPLITS = {1: "omni_1A", 2: "omni_1B", 3: "omni_2A", 4: 'omni_3A'}


def main(args):
    root = args.root
    target_split = args.target_split
    view_size = args.view_size
    crop_rate = args.crop_rate

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
            visualize_one_vid(scene_dir_path, scene, view_size, crop_rate)


def visualize_one_vid(img_dir_path, vid_name, view_size, crop_rate):
    print(f"\n--- Processing {vid_name}")
    imgs = [x for x in sorted(os.listdir(img_dir_path)) if x.endswith(".jpg")]
    for i, img_name in enumerate(imgs):
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        if crop_rate is not None:
            img = img[int(h * crop_rate[1][0]): int(h * crop_rate[1][1]),
                      int(w * crop_rate[0][0]): int(w * crop_rate[0][1])]
        if view_size is not None and [w, h] != view_size:
            img = letterbox(img, view_size[::-1], auto=False)[0]
        info = f"{vid_name}: {i + 1} / {len(imgs)}"
        plot_info(img, info)
        cv2.imshow("img", img)
        cv2.waitKey(0)


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


def plot_info(img, info, font_size=2, font_thickness=2):
    label_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] * 2), [0, 0, 0], -1)
    cv2.putText(img, info, (5, int(label_size[1] * 1.5))
                , cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/PIROPO"
    parser.add_argument("--root", type=str, default=root)

    # 1: omni_1A, 2: omni_1B, 3: omni_2A, 4: omni_3A
    target_split = 1
    target_split = None
    parser.add_argument("--target-split", type=str, default=target_split)

    view_size = [1280, 720]
    parser.add_argument("--view-size", type=int, default=view_size)

    crop_rate = [(0.2, 0.8), (0.2, 0.8)]
    parser.add_argument("--crop-rate", type=float, default=crop_rate)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
