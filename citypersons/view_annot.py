# View annotations of "cityscapes-citypersons" dataset
import argparse
import os
import json

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
    target_split = args.target_split
    target_city = args.target_city
    view_size = args.view_size
    view_thr = args.view_thr
    area_thr = args.area_thr
    clip_bbox = args.clip_bbox

    img_root = os.path.join(img_root, "leftImg8bit")
    annot_root = os.path.join(annot_root, "gtBboxCityPersons")
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
            visualize_one_vid(city_dir_path, f"{split}-{city}",
                              annot_dir_path, view_size, view_thr, area_thr, clip_bbox)


def visualize_one_vid(img_dir_path, vid_name, annot_dir_path, view_size, view_thr, area_thr, clip_bbox):
    print(f"\n--- Processing {vid_name}")
    imgs = [x for x in sorted(os.listdir(img_dir_path)) if x.endswith(".png")]
    annots = [x for x in sorted(os.listdir(annot_dir_path)) if x.endswith(".json")]
    for i, (img_name, annot_name) in enumerate(zip(imgs, annots)):
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        is_resized = False
        if view_size is not None and [w, h] != view_size:
            img, ratio, (dw, dh) = letterbox(img, view_size[::-1], auto=False)
            is_resized = True

        annot_path = os.path.join(annot_dir_path, annot_name)
        with open(annot_path) as f:
            annot = json.load(f)
        bboxes = annot["objects"]
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
                    xyxy = xyxy2xyxyc(xyxy, view_size[0], view_size[1])
                else:
                    xyxy = xyxy2xyxyc(xyxy, w, h)
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if area < area_thr:
                continue
            label = bbox["label"]
            if label == "ignore":
                color = [0, 0, 255]
            else:
                color = [0, 255, 255]
            img = cv2.rectangle(img, xyxy[:2], xyxy[2:], color, 1)

        info = f"{vid_name}: {i + 1} / {len(imgs)}"
        plot_info(img, info)
        cv2.imshow("img", img)
        cv2.waitKey(0)


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

    img_root = "/media/daton/Data/datasets/cityscapes/leftImg8bit_trainvaltest"
    parser.add_argument("--img-root", type=str, default=img_root)

    annot_root = "/media/daton/Data/datasets/cityscapes/gtBbox_cityPersons_trainval"
    parser.add_argument("--annot-root", type=str, default=annot_root)

    # 1: train, 2: val
    target_split = [1]
    target_split = None
    parser.add_argument("--target-split", type=str, default=target_split)

    # Different by split
    target_city = [1, 3, 5, 7]
    target_city = None
    parser.add_argument("--target-city", type=str, default=target_city)

    view_size = [1280, 720]
    parser.add_argument("--view-size", type=int, default=view_size)

    view_thr = 0.3
    parser.add_argument("--view-thr", type=float, default=view_thr)

    area_thr = 200
    parser.add_argument("--area-thr", type=int, default=area_thr)

    parser.add_argument("--clip-bbox", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
