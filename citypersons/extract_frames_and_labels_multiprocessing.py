# Save images and labels
# total frame: 2777, total obj: 18892 with area threshold 250, visibility threshold 0.3,
# Elapsed time: 232.52 -> 46.02(by multiprocessing)
import argparse
import os
import re
import glob
import json
import time
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np


SPLITS = {1: "train", 2: "val"}
CITIES = {"train": {1: "aachen", 2: "bochum", 3: "bremen", 4: "cologne", 5: "darmstadt", 6: "dusseldorf",
                    7: "erfurt", 8: "hamburg", 9: "hanover", 10: "jena", 11: "krefeld", 12: "monchengladbach",
                    13: "strasbourg", 14: "stuttgart", 15: "tubingen", 16: "ulm", 17: "weimar", 18: "zurich"},
          "val": {1: "frankfurt", 2: "lindau", 3: "munster"}}


def main(args):
    img_root = args.img_root
    annot_root = args.annot_root
    save_dir = args.save_dir
    run_name = args.run_name
    target_split = args.target_split
    target_city = args.target_city
    target_size = args.target_size
    view_thr = args.view_thr
    area_thr = args.area_thr
    clip_bbox = args.clip_bbox
    skip_ignored = args.skip_ignored
    save = args.save
    num_workers = args.num_workers

    save_dir = increment_path(Path(save_dir) / run_name, exist_ok=False)
    img_save_dir = str(save_dir / "images")
    gt_save_dir = str(save_dir / "labels")
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)
        os.mkdir(img_save_dir)
        os.mkdir(gt_save_dir)

    img_root = os.path.join(img_root, "leftImg8bit")
    annot_root = os.path.join(annot_root, "gtBboxCityPersons")
    total_paths = []

    ts = time.time()
    splits = [x for x in os.listdir(img_root)
              if os.path.isdir(os.path.join(img_root, x)) and x in SPLITS.values()]
    if target_split is not None:
        assert all(x in SPLITS for x in target_split), \
            f"Some elements of {target_split} are not in {SPLITS}"
        splits = [SPLITS[x] for x in target_split]

    for split in splits:
        split_dir_path = os.path.join(img_root, split)
        cities = [x for x in os.listdir(split_dir_path)
                  if os.path.isdir(os.path.join(split_dir_path, x)) and x in CITIES[split].values()]
        if target_city is not None:
            assert all(x in CITIES[split] for x in target_city), \
                f"Some elements of {target_city} are not in {CITIES[split]}"
            cities = [CITIES[split][x] for x in target_city]

        for city in cities:
            city_dir_path = os.path.join(split_dir_path, city)
            annot_dir_path = os.path.join(annot_root, split, city)
            total_paths.append((city_dir_path, f"{split}-{city}", annot_dir_path))

    pool = mp.Pool(num_workers)
    pool.map(
        extract_frames_and_labels,
        zip(total_paths,
            len(total_paths) * [target_size],
            len(total_paths) * [view_thr],
            len(total_paths) * [area_thr],
            len(total_paths) * [clip_bbox],
            len(total_paths) * [skip_ignored],
            len(total_paths) * [img_save_dir],
            len(total_paths) * [gt_save_dir],
            len(total_paths) * [save])
    )
    pool.close()
    pool.join()
    te = time.time()
    total_save_cnt = len([x for x in os.listdir(img_save_dir) if x.endswith(".png")])
    print(f"\n--- Total save count: {total_save_cnt}, Elapsed time: {te - ts:.2f}s")


def extract_frames_and_labels(input_item):
    (img_dir_path, vid_name, annot_dir_path), \
    target_size, view_thr, area_thr, clip_bbox, skip_ignored, img_save_dir, gt_save_dir, save = input_item
    imgs = [x for x in sorted(os.listdir(img_dir_path)) if x.endswith(".png")]
    annots = [x for x in sorted(os.listdir(annot_dir_path)) if x.endswith(".json")]
    frame_cnt = 0
    obj_cnt = 0
    for i, (img_name, annot_name) in enumerate(zip(imgs, annots)):
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        is_resized = False
        if target_size is not None and [w, h] != target_size:
            img, ratio, (dw, dh) = letterbox(img, target_size[::-1], auto=False)
            is_resized = True

        annot_path = os.path.join(annot_dir_path, annot_name)
        with open(annot_path) as f:
            annot = json.load(f)
        bboxes = annot["objects"]
        txt = ""
        for bbox in bboxes:
            xywh = bbox["bbox"]
            xywhv = bbox["bboxVis"]
            vis_rate = get_vis_rate(xywh, xywhv)
            if vis_rate < view_thr:
                continue
            xyxy = xywh2xyxy(xywh)
            if is_resized:
                xyxy = resize_xyxy(xyxy, ratio, dw, dh)
            if clip_bbox:
                if is_resized:
                    xyxy = xyxy2xyxyc(xyxy, target_size[0], target_size[1])
                else:
                    xyxy = xyxy2xyxyc(xyxy, w, h)
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if area < area_thr:
                continue
            label = bbox["label"]
            if skip_ignored and label == "ignore":
                continue
            if target_size is not None:
                cpwhn = xyxy2cpwhn(xyxy, target_size[0], target_size[1])
            else:
                cpwhn = xyxy2cpwhn(xyxy, w, h)
            txt += f"0 {cpwhn[0]} {cpwhn[1]} {cpwhn[2]} {cpwhn[3]}\n"
            obj_cnt += 1
        if txt != "":
            if save:
                img_save_path = os.path.join(img_save_dir, img_name)
                cv2.imwrite(img_save_path, img)
                gt_save_path = os.path.join(gt_save_dir, ".".join(img_name.split(".")[:-1]) + ".txt")
                with open(gt_save_path, "w") as gt:
                    gt.write(txt)
            frame_cnt += 1
    print(f"\tframe: {frame_cnt}, obj: {obj_cnt} from {vid_name}")


def get_vis_rate(xywh1, xywh2):
    area1 = xywh1[2] * xywh1[3]
    area2 = xywh1[2] * xywh2[3]
    return area2 / area1


def xywh2xyxy(xywh):
    xyxy = [xywh[0],
            xywh[1],
            xywh[0] + xywh[2],
            xywh[1] + xywh[3]]
    return xyxy


def resize_xyxy(xyxy, ratio, dw, dh):
    xyxy = [int(xyxy[0] * ratio[0] + dw),
            int(xyxy[1] * ratio[1] + dh),
            int(xyxy[2] * ratio[0] + dw),
            int(xyxy[3] * ratio[1] + dh)]
    return xyxy


def xyxy2xyxyc(xyxy, w, h):
    xyxy = [min(max(0, xyxy[0]), w),
            min(max(0, xyxy[1]), h),
            min(max(0, xyxy[2]), w),
            min(max(0, xyxy[3]), h)]
    return xyxy


def xyxy2cpwhn(xyxy, img_w, img_h):
    cpwhn = [round((xyxy[0] + xyxy[2]) / 2 / img_w, 6),
             round((xyxy[1] + xyxy[3]) / 2 / img_h, 6),
             round((xyxy[2] - xyxy[0]) / img_w, 6),
             round((xyxy[3] - xyxy[1]) / img_h, 6)]
    return cpwhn


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

    img_root = "/media/daton/Data/datasets/cityscapes/leftImg8bit_trainvaltest"
    parser.add_argument("--img-root", type=str, default=img_root)

    annot_root = "/media/daton/Data/datasets/cityscapes/gtBbox_cityPersons_trainval"
    parser.add_argument("--annot-root", type=str, default=annot_root)

    save_dir = "/media/daton/Data/datasets/cityscapes/extracted_frames"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    run_name = "exp"
    parser.add_argument("--run-name", type=str, default=run_name)

    # 1: train, 2: val
    target_split = [1]
    target_split = None
    parser.add_argument("--target-split", type=str, default=target_split)

    # Different by split
    target_city = [1, 3, 5, 7]
    target_city = None
    parser.add_argument("--target-city", type=str, default=target_city)

    target_size = [1280, 720]
    parser.add_argument("--target-size", type=int, default=target_size)

    view_thr = 0.3
    parser.add_argument("--view-thr", type=float, default=view_thr)

    area_thr = 250
    parser.add_argument("--area-thr", type=int, default=area_thr)

    parser.add_argument("--clip-bbox", action="store_true", default=True)
    parser.add_argument("--skip-ignored", action="store_true", default=True)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=8)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
