# YOLOv5 formatted labels (cls, cpwhn) -> COCO format
# crowdhuman train: 15000 images and 33xxxx samples
# crowdhuman val: 4370 images and 99481 samples
# citypersons: 2777 images and 18892 samples
# MOT17: 108 images and 2092 samples
# MOT20: 38 images and 5243 samples
import argparse
import json
import os
import time

import cv2
from tqdm import tqdm


def main(args):
    img_dir = args.img_dir
    yolov5_dir = args.yolov5_dir
    save_dir = args.save_dir
    out_name = args.out_name
    exp_format = args.exp_format
    save = args.save

    if save and not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, out_name)

    img_cnt = 0
    ann_cnt = 0
    out = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}
    annots = [x for x in os.listdir(yolov5_dir) if x.endswith(".txt")]
    print(f"\n--- Processing {yolov5_dir}")
    time.sleep(0.1)
    for annot_name in tqdm(annots):
        img_name = annot_name.replace(".txt", f".{exp_format}")
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        img_cnt += 1
        img_info = {"file_name": img_name,
                    "id": img_cnt,
                    "height": height,
                    "width": width}
        out["images"].append(img_info)

        annot_path = os.path.join(yolov5_dir, annot_name)
        with open(annot_path) as a:
            annot = a.readlines()
        for bbox in annot:
            cls, cxn, cyn, wn, hn = bbox.split()
            xywh = cpwhn2xywh(list(map(float, (cxn, cyn, wn, hn))), width, height)
            ann_cnt += 1
            ann_info = {"id": ann_cnt,
                        "category_id": 1,
                        "image_id": img_cnt,
                        "bbox": xywh,
                        "area": xywh[2] * xywh[3],
                        "iscrowd": 0}
            out["annotations"].append(ann_info)
    time.sleep(0.1)
    print('\t{} images and {} samples'.format(len(out['images']), len(out['annotations'])))
    if save:
        json.dump(out, open(save_path, 'w'))


def cpwhn2xywh(cpwh, w, h):
    xywh = [int((cpwh[0] - cpwh[2] / 2) * w),
            int((cpwh[1] - cpwh[3] / 2) * h),
            int(cpwh[2] * w),
            int(cpwh[3] * h)]
    return xywh


def parse_args():
    parser = argparse.ArgumentParser()

    img_dir = "/media/daton/Data/datasets/crowdhuman/extracted_frames/train/images"
    img_dir = "/media/daton/Data/datasets/crowdhuman/extracted_frames/val/images"
    img_dir = "/media/daton/Data/datasets/cityscapes/extracted_frames/exp2/images"
    img_dir = "/media/daton/Data/datasets/mot/extracted_frames/MOT17/images"
    img_dir = "/media/daton/Data/datasets/mot/extracted_frames/MOT20/images"
    parser.add_argument("--img-dir", type=str, default=img_dir)

    yolov5_dir = "/media/daton/Data/datasets/crowdhuman/extracted_frames/train/labels"
    yolov5_dir = "/media/daton/Data/datasets/crowdhuman/extracted_frames/val/labels"
    yolov5_dir = "/media/daton/Data/datasets/cityscapes/extracted_frames/exp2/labels"
    yolov5_dir = "/media/daton/Data/datasets/mot/extracted_frames/MOT17/labels"
    yolov5_dir = "/media/daton/Data/datasets/mot/extracted_frames/MOT20/labels"
    parser.add_argument("--yolov5-dir", type=str, default=yolov5_dir)

    save_dir = "/media/daton/Data/datasets/general_person_dataset/coco_annotations"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    out_name = "MOT20.json"
    parser.add_argument("--out-name", type=str, default=out_name)

    exp_format = "jpg"
    parser.add_argument("--exp-format", type=str, default=exp_format)

    parser.add_argument("--save", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
