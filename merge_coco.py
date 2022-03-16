# Merge coco dataset labels
# train: Total 20237 images, Total 372078 samples
# val: Total 5367 images, Total 102181 samples
import argparse
import os
import json
import time
from pathlib import Path

from tqdm import tqdm


def main(args):
    coco_path_list = args.coco_path_list
    save_dir = args.save_dir
    out_name = args.out_name
    save = args.save

    if save and not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, out_name)

    img_cnt = 1
    ann_cnt = 1
    out = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}

    for coco_path in coco_path_list:
        print(f"\n--- Processing {Path(coco_path).name}")
        with open(coco_path) as f:
            annot = json.load(f)
        imgs_info = annot["images"]
        print(f"\tcopying images info")
        time.sleep(0.1)
        img_id_dict = {}
        for tmp_img_info in tqdm(imgs_info):
            img_id_dict[tmp_img_info["id"]] = img_cnt
            img_info = {"file_name": tmp_img_info["file_name"],
                        "id": img_cnt,
                        "height": tmp_img_info["height"],
                        "width": tmp_img_info["width"]}
            out["images"].append(img_info)
            img_cnt += 1
        annots_info = annot["annotations"]
        print(f"\tcopying annotations info")
        time.sleep(0.1)
        for tmp_ann_info in tqdm(annots_info):
            ann_info = {"id": ann_cnt,
                        "category_id": 1,
                        "image_id": img_id_dict[tmp_ann_info["image_id"]],
                        "bbox": tmp_ann_info["bbox"],
                        "area": tmp_ann_info["area"],
                        "iscrowd": tmp_ann_info["iscrowd"]}
            out["annotations"].append(ann_info)
            ann_cnt += 1
        print(f"{len(imgs_info)} images and {len(annots_info)} samples")
    print(f"\n Total {len(out['images'])} images, Total {len(out['annotations'])} samples")

    if save:
        json.dump(out, open(save_path, "w"))


def parse_args():
    parser = argparse.ArgumentParser()

    coco_path_list = [
        "/media/daton/Data/datasets/general_person_dataset/coco_annotations/citypersons.json",
        "/media/daton/Data/datasets/general_person_dataset/coco_annotations/crowdhuman_train.json",
        "/media/daton/Data/datasets/general_person_dataset/coco_annotations/MOT17.json",
        "/media/daton/Data/datasets/general_person_dataset/coco_annotations/MOT20.json",
        "/media/daton/Data/datasets/general_person_dataset/coco_annotations/custom_train.json"
    ]
    '''coco_path_list = [
        "/media/daton/Data/datasets/general_person_dataset/coco_annotations/crowdhuman_val.json",
        "/media/daton/Data/datasets/general_person_dataset/coco_annotations/custom_val.json"
    ]'''
    parser.add_argument("--coco-path-list", nargs="+", type=str, default=coco_path_list)

    save_dir = "/media/daton/Data/datasets/general_person_dataset/coco_annotations"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    out_name = "train.json"
    parser.add_argument("--out-name", type=str, default=out_name)

    parser.add_argument("--save", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
