# Save images and labels (yolov5 style: class, cx, cy, width, height)
import argparse
import os
import re
import glob
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SELECTIONS = {1: "train", 2: "val"}
CLASSES = {1: "fbox", 2: "vbox", 3: "hbox"}


def main(args):
    root = args.root
    save_dir = args.save_dir
    run_name = args.run_name
    target_select = args.target_select
    target_split = args.target_split
    target_class = args.target_class
    target_size = args.target_size
    crop_bbox = args.crop_bbox
    skip_ignored = args.skip_ignored
    save = args.save

    save_dir = increment_path(Path(save_dir) / run_name, exist_ok=False)
    img_save_dir = str(save_dir / "images")
    gt_save_dir = str(save_dir / "labels")
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)
        os.mkdir(img_save_dir)
        os.mkdir(gt_save_dir)

    selects_dict = {}
    img_dirs = [x for x in os.listdir(root)
                if os.path.isdir(os.path.join(root, x)) and any(v in x for v in SELECTIONS.values())]
    if target_select is not None:
        assert all(x in SELECTIONS for x in target_select), f"Some elements of {target_select} are not in {SELECTIONS}"
        tmp_select = target_select
    else:
        tmp_select = SELECTIONS
    for select in tmp_select:
        select_dirs = [x for x in img_dirs if SELECTIONS[select] in x]
        selects_dict[SELECTIONS[select]] = select_dirs
    print(selects_dict)

    for select, splits in selects_dict.items():
        if select == "train" and target_split is not None:
            splits = [x for x in splits if int(x[-2:]) in target_split]
            assert len(splits) == len(target_split), f"Some elements of {target_split} are wrong!"
        img_paths_dict = {}
        for split in splits:
            split_dir_path = os.path.join(root, split, "Images")
            for img_name in [x for x in os.listdir(split_dir_path) if os.path.isfile(os.path.join(split_dir_path, x))]:
                img_paths_dict[img_name] = os.path.join(split_dir_path, img_name)

        assert all(x in CLASSES for x in target_class), f"Some elements of {target_class} are wrong!"
        annot_path = os.path.join(root, f"annotation_{select}.odgt")
        visualize_images(img_paths_dict, annot_path, img_save_dir, gt_save_dir,
                         target_class, target_size, crop_bbox, skip_ignored, save)


def visualize_images(img_paths_dict, annot_path, img_save_dir, gt_save_dir,
                     target_class, target_size, crop_bbox, skip_ignored, save):
    with open(annot_path) as f:
        data = f.readlines()
        for d in tqdm(data):
            annot = eval(d)
            img_name = f"{annot['ID']}.jpg"
            if img_name not in img_paths_dict:
                continue
            img_path = img_paths_dict[img_name]
            img = cv2.imread(img_path)
            is_resized = False
            if target_size is not None and img.shape[:2] != target_size[::-1]:
                img, ratio, (dw, dh) = letterbox(img, target_size[::-1], auto=False)
                is_resized = True
            txt = ""
            for gtbox in annot["gtboxes"]:
                extra = gtbox["extra"]
                if "ignore" in extra:
                    ignore = bool(extra["ignore"])
                else:
                    ignore = False
                if skip_ignored and ignore:
                    continue
                bboxes = {c: gtbox[CLASSES[c]] for c in target_class}

                for cls, bbox in bboxes.items():
                    if is_resized:
                        bbox = resize_xywh(bbox, ratio, dw, dh)
                    if crop_bbox:
                        h, w, _ = img.shape
                        xyxy = xywh2xyxyc(bbox, w, h)
                    else:
                        xyxy = xywh2xyxy(bbox)
                    cpwhn = xyxy2cpwhn(xyxy, w, h)
                    txt += f"{cls - 1} {cpwhn[0]} {cpwhn[1]} {cpwhn[2]} {cpwhn[3]}\n"
            if save:
                img_save_path = os.path.join(img_save_dir, img_name)
                gt_save_path = os.path.join(gt_save_dir, ".".join(img_name.split(".")[:-1]) + ".txt")
                cv2.imwrite(img_save_path, img)
                with open(gt_save_path, "w") as gt:
                    gt.write(txt)


def xyxy2cpwhn(xyxy, img_w, img_h):
    cpwhn = [round((xyxy[0] + xyxy[2]) / 2 / img_w, 6),
             round((xyxy[1] + xyxy[3]) / 2 / img_h, 6),
             round((xyxy[2] - xyxy[0]) / img_w, 6),
             round((xyxy[3] - xyxy[1]) / img_h, 6)]
    return cpwhn

def xywh2xyxyc(xywh, w, h):
    xyxy = [min(max(0, xywh[0]), w),
            min(max(0, xywh[1]), h),
            min(max(0, xywh[0] + xywh[2]), w),
            min(max(0, xywh[1] + xywh[3]), h)]
    return xyxy


def xywh2xyxy(xywh):
    xyxy = [xywh[0],
            xywh[1],
            xywh[0] + xywh[2],
            xywh[1] + xywh[3]]
    return xyxy


def resize_xywh(xywh, ratio, dw, dh):
    xywh = [int(xywh[0] * ratio[0] + dw),
            int(xywh[1] * ratio[1] + dh),
            int(xywh[2] * ratio[0]),
            int(xywh[3] * ratio[1])]
    return xywh


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

    root = "/media/daton/Data/datasets/crowdhuman"
    parser.add_argument("--root", type=str, default=root)

    save_dir = "/media/daton/Data/datasets/crowdhuman/extracted_frames"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    run_name = "trainval"
    parser.add_argument("--run-name", type=str, default=run_name)

    # 1: train, 2: valid
    target_select = [2]
    target_select = None
    parser.add_argument("--target-select", type=int, default=target_select)

    # train: [1, 2, 3], val: None
    target_split = [1, 2]
    target_split = None
    parser.add_argument("--target-split", type=int, default=target_split)

    # 1: fbox, 2: vbox, 3: hbox
    target_class = [1]
    # target_class = None
    parser.add_argument("--target-class", type=int, default=target_class)

    target_size = [1280, 720]
    parser.add_argument("--target-size", type=int, default=target_size)

    parser.add_argument("--crop-bbox", action="store_true", default=True)
    parser.add_argument("--skip-ignored", action="store_true", default=True)
    parser.add_argument("--save", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
