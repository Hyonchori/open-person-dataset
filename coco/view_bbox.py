# View bounding box annotations from COCO dataset
import argparse
import os

import cv2
from pycocotools.coco import COCO


SELECTION = {1: "train2017", 2: "val2017"}


def main(args):
    root = args.root
    select = args.select
    target_cls = args.target_class
    view_size = args.view_size
    hide_info = args.hide_info
    hide_label = args.hide_labels

    target_selection = SELECTION[select]
    annot_dir = os.path.join(root, "annotations_trainval2017", "annotations")
    annot_path = os.path.join(annot_dir, f"captions_{target_selection}.json")
    annot = COCO(annot_path)
    print(annot.getImgIds(catIds=9))

    img_dir = os.path.join(root, target_selection)





def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/coco"
    parser.add_argument("--root", type=str, default=root)
    parser.add_argument("--select", type=int, default=1)  # 1: train2017, 2: val2017

    target_cls = [1]
    target_cls = None
    parser.add_argument("--target-class", type=int, default=target_cls)

    view_size = [1280, 720]
    parser.add_argument("--view-size", type=int, default=view_size)

    parser.add_argument("--hide_info", action="store_true", default=False)
    parser.add_argument("--hide-labels", action="store_true", default=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
