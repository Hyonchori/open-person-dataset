import argparse
import os
import json

import numpy as np


def main(args):
    root = args.root
    max_num_per_cam = args.max_num_per_cam
    save_interval = args.save_interval
    box_ratio = args.box_ratio
    execute = args.execute
    target_action = args.target_action

    annot_2d_root = os.path.join(root, "annotation", "Annotation_2D_tar", "2D")
    image_root = os.path.join(root, "이미지")
    save_cnt = 0
    remove_cnt = 0

    action_dict = {int(x.split("_")[-1]): x for x in os.listdir(image_root)
                   if os.path.isdir(os.path.join(image_root, x)) and int(x.split("_")[-1]) in target_action}
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
                    save_cnt, remove_cnt = save_only_target(cam_dir_path, target_annot, max_num_per_cam, save_interval,
                                                            box_ratio, execute, save_cnt, remove_cnt)
    print(f"\nTotal saved images: {save_cnt}")
    print(f"Total removed images: {remove_cnt}")


def save_only_target(img_dir_path, annot, max_num_per_cam, save_interval, box_ratio, execute,
                     save_cnt, remove_cnt):
    imgs = sorted(os.listdir(img_dir_path), reverse=True)
    cnt = 0
    tmp_num = None
    save_imgs = []
    target_indices = [i for i, x in enumerate(annot["images"]) if x["img_path"].split("/")[-1] in imgs]
    for img_name, target_idx in zip(imgs, target_indices):
        img_num = int(img_name.split(".")[0].split("_")[-1])
        bbox = annot["annotations"][target_idx]["bbox"]
        if None in bbox:
            continue
        width, height = get_width_height(bbox)
        if width == 0 or height == 0:
            continue
        ratio = height / width
        if cnt <= max_num_per_cam and \
            ratio <= box_ratio and \
            (tmp_num is None or tmp_num - img_num >= save_interval):
            save_imgs.append(img_name)
            tmp_num = img_num
            cnt += 1
            if cnt == max_num_per_cam:
                break
    removed_imgs = list(set(imgs) - set(save_imgs))
    if execute:
        print(f"\n--- Processing {img_dir_path}")
        for img_name in removed_imgs:
            img_path = os.path.join(img_dir_path, img_name)
            os.remove(img_path)
        print(f"\tdelete {len(removed_imgs)} images")
    save_cnt += len(save_imgs)
    remove_cnt += len(removed_imgs)
    return save_cnt, remove_cnt


def get_width_height(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/사람동작 영상"
    parser.add_argument("--root", type=str, default=root)

    parser.add_argument("--max-num-per-cam", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--box-ratio", type=float, default=0.8)  # height / width
    parser.add_argument("--execute", type=bool, default=False)
    parser.add_argument("--target-action", type=int, default=[3, 6, 14])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
