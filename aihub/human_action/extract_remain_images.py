import argparse
import os
import json
import time

import cv2
from tqdm import tqdm


def main(args):
    root = args.root
    out_dir = args.out_dir
    prefix = args.prefix
    execute = args.execute
    resize = args.resize
    max_num_per_case = args.max_num_per_case
    img_out_dir = os.path.join(out_dir, "images")
    if not os.path.isdir(img_out_dir) and execute:
        os.makedirs(img_out_dir)
    label_out_dir = os.path.join(out_dir, "labels")
    if not os.path.isdir(label_out_dir) and execute:
        os.makedirs(label_out_dir)

    annot_2d_root = os.path.join(root, "annotation", "Annotation_2D_tar", "2D")
    image_root = os.path.join(root, "이미지")

    cnt = 0
    ts = time.time()
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
            time.sleep(0.5)
            print(f"\n---Processing {num_dir_path}")
            time.sleep(0.5)
            for case_key in tqdm(case_dict):
                cam_dict = {int(x.split("-")[-1][1:]): x for x in case_dict[case_key]}
                cam_cnt = 0
                for cam_key in cam_dict:
                    cam_dir_path = os.path.join(num_dir_path, cam_dict[cam_key])
                    if len(os.listdir(cam_dir_path)) == 0:
                        continue
                    else:
                        if cam_cnt >= max_num_per_case:
                            break
                        else:
                            cam_cnt += 1
                    target_annot_dir = os.path.join(annot_2d_root, cam_dict[cam_key].split("_")[0])
                    target_annot_path = os.path.join(target_annot_dir, f"{cam_dict[cam_key]}_2D.json")
                    with open(target_annot_path) as f:
                        target_annot = json.load(f)
                    cnt += extract_imglabels_to_out_dir(cam_dir_path, target_annot, img_out_dir, label_out_dir, prefix,
                                                        resize, execute)
    te = time.time()
    print(f"\nTotal images: {cnt}")  # 10593 -> 3625 -> 2410
    print(f"Elapsed time: {te - ts:.2f}s")


def extract_imglabels_to_out_dir(img_dir_path, annot, img_out_dir, label_out_dir, prefix, resize, execute):
    imgs = sorted(os.listdir(img_dir_path))
    target_indices = [i for i, x in enumerate(annot["images"]) if x["img_path"].split("/")[-1] in imgs]
    cnt = 0
    for img_name, target_idx in zip(imgs, target_indices):
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        bbox = annot["annotations"][target_idx]["bbox"]
        cpwhn = xyxy2cpwhn(bbox, w, h)
        if None in bbox:
            continue
        label = f"0 {cpwhn[0]} {cpwhn[1]} {cpwhn[2]} {cpwhn[3]}"

        if execute:
            label_path = os.path.join(label_out_dir, img_name.replace(".jpg", ".txt"))
            with open(label_path, "w") as f:
                f.write(label)
            out_img_path = os.path.join(img_out_dir, f"{prefix}_{img_name}")
            img = cv2.resize(img, dsize=(resize[0], resize[1]))
            cv2.imwrite(out_img_path, img)
        cnt += 1
        cv2.circle(img, (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)), 4, [0, 0, 255], -1)
    return cnt


def xyxy2cpwhn(xyxy, w, h):
    cpwhn = [round((xyxy[0] + xyxy[2]) / 2 / w, 6),
             round((xyxy[1] + xyxy[3]) / 2 / h, 6),
             round((xyxy[2] - xyxy[0]) / w, 6),
             round((xyxy[3] - xyxy[1]) / h, 6)]
    return cpwhn


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/사람동작 영상"
    parser.add_argument("--root", type=str, default=root)

    out_dir = "/media/daton/Data/datasets/aihub/human_behavior_images"
    parser.add_argument("--out-dir", type=str, default=out_dir)

    prefix = "human_action"
    parser.add_argument("--prefix", type=str, default=prefix)
    parser.add_argument("--execute", type=bool, default=True)
    parser.add_argument("--resize", type=int, default=[1280, 720])
    parser.add_argument("--max-num-per-case", type=int, default=3)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

