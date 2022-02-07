import argparse
import os
import json
import time

import cv2
from tqdm import tqdm


def main(args):
    root = args.root
    out_dir = args.out_dir
    out_name = args.out_name

    out_path = os.path.join(out_dir, out_name) if out_dir is not None else os.path.join(root, out_name)
    out = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}
    img_cnt = 0
    ann_cnt = 0

    splits = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
    for split in splits:
        split_dir_path = os.path.join(root, split)
        actions = [x for x in os.listdir(split_dir_path) if os.path.isdir(os.path.join(split_dir_path, x))]
        for action in actions:
            action_dir_path = os.path.join(split_dir_path, action)
            case_dict = {int(x.split("_")[-1]): x for x in os.listdir(action_dir_path)
                         if os.path.isdir(os.path.join(action_dir_path, x)) and "원천" in x}
            annot_dict = {k: v.replace("원천", "라벨") for k, v in case_dict.items()}
            for case in case_dict:
                case_dir_path = os.path.join(action_dir_path, case_dict[case])
                scenes = [x for x in os.listdir(case_dir_path) if os.path.isdir(os.path.join(case_dir_path, x))]
                time.sleep(0.5)
                print(f"\n--- Processing {case_dir_path}")
                time.sleep(0.5)
                for scene in tqdm(scenes):
                    scene_dir_path = os.path.join(case_dir_path, scene)
                    target_annot_path = os.path.join(action_dir_path, annot_dict[case], f"annotation_{scene}.json")
                    with open(target_annot_path) as f:
                        target_annot = json.load(f)
                    img_cnt, ann_cnt = get_tmp_info(scene_dir_path, target_annot, img_cnt, ann_cnt, out)
    print(f"Summary: {len(out['images'])} images, {len(out['annotations'])} samples")
    print(f"save_path: {out_path}")
    json.dump(out, open(out_path, "w"))


def get_tmp_info(img_dir_path, annots, img_cnt, ann_cnt, coco):
    imgs = sorted(os.listdir(img_dir_path), reverse=True)
    target_indices = [(i, x) for i, x in enumerate(annots["frames"]) if x["image"] in imgs]
    for img_name, (target_idx, img_info) in zip(imgs, target_indices):
        img_cnt += 1
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        coco_img_info = {"file_name": img_name,
                         "id": img_cnt,
                         "height": h,
                         "width": w}
        coco["images"].append(coco_img_info)
        bboxes = img_info["annotations"]
        for bbox in bboxes:
            bbox = bbox["label"]
            xywh = bbox2xywh(bbox)
            ann_cnt += 1
            coco_ann_info = {"id": ann_cnt,
                             "category_id": 1,
                             "image_id": img_cnt,
                             "bbox": xywh,
                             "area": xywh[2] * xywh[3],
                             "iscrowd": 0}
            coco["annotations"].append(coco_ann_info)
    return img_cnt, ann_cnt


def bbox2xywh(bbox):
    xywh = [bbox["x"],
            bbox["y"],
            bbox["width"],
            bbox["height"]]
    return xywh

def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/지하철 역사 내 CCTV 이상행동 영상"
    parser.add_argument("--root", type=str, default=root)

    out_dir = None
    parser.add_argument("--out-dir", type=str, default=out_dir)
    parser.add_argument("--out-name", type=str, default="subway_cctv_coco.json")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)