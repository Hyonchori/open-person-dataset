import argparse
import os

from pycocotools.coco import COCO


def main(args):
    coco_path = args.coco_path
    save_dir = args.save_dir

    coco_annot = COCO(coco_path)
    img_ids = coco_annot.getImgIds()
    ann_ids = coco_annot.getAnnIds(imgIds=5)
    print(ann_ids)



def parse_args():
    parser = argparse.ArgumentParser()

    coco_path = "/media/daton/Data/datasets/general_person_dataset/coco_annotations/train.json"
    parser.add_argument("--coco-path", type=str, default=coco_path)

    save_dir = "/media/daton/Data/datasets/general_person_dataset/labels"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
