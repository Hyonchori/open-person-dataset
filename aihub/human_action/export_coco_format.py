import argparse
import os
import json


def main(args):
    root = args.root
    out_dir = args.out_dir
    out_name = args.out_name

    annot_2d_root = os.path.join(root, "annotation", "Annotation_2D_tar", "2D")
    image_root = os.path.join(root, "이미지")
    out_path = os.path.join(out_dir, out_name) if out_dir is not None else os.path.join(root, out_name)
    out = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}
    img_cnt = 0
    ann_cnt = 0

    action_dict = {int(x.split("_")[-1]): x for x in os.listdir(image_root)
                   if os.path.isdir(os.path.join(image_root, x))}
    for action_key in action_dict:
        action_dir_path = os.path.join(image_root, action_dict[action_key])
        num_dict = {int(x.split("-")[-1]): x for x in os.listdir(action_dir_path)
                    if os.path.isdir(os.path.join(action_dir_path, x))}
        for num_key in num_dict:
            num_dir_path = os.path.join(action_dir_path, num_dict[num_key])
            while len(os.listdir(num_dir_path)) == 1:
                num_dir_path = os.path.join(num_dir_path, os.listdir(num_dir_path)[0])
            case_dict = {}
            for case in os.listdir(num_dir_path):
                if not os.path.isdir(os.path.join(num_dir_path, case)):
                    continue
                case_idx = int(case.split("_")[-1].split("-")[0])
                if case_idx in case_dict:
                    case_dict[case_idx].append(case)
                else:
                    case_dict[case_idx] = [case]
            for case_key in case_dict:
                cam_dict = {int(x.split("-")[-1][1:]): x for x in case_dict[case_key]}
                for cam_key in cam_dict:
                    cam_dir_path = os.path.join(num_dir_path, cam_dict[cam_key])
                    target_annot_dir = os.path.join(annot_2d_root, cam_dict[cam_key].split("_")[0])
                    target_annot_path = os.path.join(target_annot_dir, f"{cam_dict[cam_key]}_2D.json")
                    with open(target_annot_path) as f:
                        target_annot = json.load(f)
                    img_cnt, ann_cnt = get_tmp_info(cam_dir_path, target_annot, img_cnt, ann_cnt, out)
    print(f"Summary: {len(out['images'])} images, {len(out['annotations'])} samples")
    print(f"save_path: {out_path}")
    json.dump(out, open(out_path, "w"))


def get_tmp_info(img_dir_path, annot, img_cnt, ann_cnt, coco):
    imgs = sorted(os.listdir(img_dir_path))
    target_img_infos = [(i, x) for i, x in enumerate(annot["images"]) if x["img_path"].split("/")[-1] in imgs]
    for img_name, (target_idx, img_info) in zip(imgs, target_img_infos):
        img_cnt += 1
        coco_img_info = {"file_name": img_name,
                         "id": img_cnt,
                         "height": img_info["height"],
                         "width": img_info["width"]}
        coco["images"].append(coco_img_info)
        xyxy = annot["annotations"][target_idx]["bbox"]
        xywh = xyxy2xywh(xyxy)
        ann_cnt += 1
        coco_ann_info = {"id": ann_cnt,
                         "category_id": 1,
                         "image_id": img_cnt,
                         "bbox": xywh,
                         "area": xywh[2] * xywh[3],
                         "iscrowd": 0}
        coco["annotations"].append(coco_ann_info)
    return img_cnt, ann_cnt


def xyxy2xywh(xyxy):
    xywh = [xyxy[0],
            xyxy[1],
            xyxy[2] - xyxy[0],
            xyxy[3] - xyxy[1]]
    return xywh

def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/사람동작 영상"
    parser.add_argument("--root", type=str, default=root)

    out_dir = None
    parser.add_argument("--out-dir", type=str, default=out_dir)
    parser.add_argument("--out-name", type=str, default="human_action_coco.json")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
