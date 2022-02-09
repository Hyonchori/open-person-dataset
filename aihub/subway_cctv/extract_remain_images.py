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

    img_out_dir = os.path.join(out_dir, "images")
    if not os.path.isdir(img_out_dir) and execute:
        os.makedirs(img_out_dir)
    label_out_dir = os.path.join(out_dir, "labels")
    if not os.path.isdir(label_out_dir) and execute:
        os.makedirs(label_out_dir)

    cnt = 0
    ts = time.time()
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
                print(f"\n---Processing {case_dir_path}")
                time.sleep(0.5)
                for scene in tqdm(scenes):
                    scene_dir_path = os.path.join(case_dir_path, scene)
                    target_annot_path = os.path.join(action_dir_path, annot_dict[case], f"annotation_{scene}.json")
                    with open(target_annot_path) as f:
                        target_annot = json.load(f)
                    cnt = extract_imglabels_to_out_dir(scene_dir_path, target_annot, img_out_dir, label_out_dir, prefix,
                                                        resize, execute, cnt)

    te = time.time()
    print(f"\nTotal images: {cnt}")  # 2012
    print(f"Elapsed time: {te - ts:.2f}s")


def extract_imglabels_to_out_dir(img_dir_path, annots, img_out_dir, label_out_dir, prefix, resize, execute, cnt):
    imgs = sorted(os.listdir(img_dir_path))
    target_indices = [i for i, x in enumerate(annots["frames"]) if x["image"] in imgs]
    for img_name, target_idx in zip(imgs, target_indices):
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        tmp_annots = annots["frames"][target_idx]["annotations"]
        label = ""
        for tmp_annot in tmp_annots:
            bbox = tmp_annot["label"]
            if None in bbox.values():
                continue
            cpwhn = bbox2cpwhn(bbox, w, h)
            label += f"0 {cpwhn[0]} {cpwhn[1]} {cpwhn[2]} {cpwhn[3]}\n"
        cnt += 1
        if execute:
            img_name = f"{prefix}_{cnt:04d}_{img_name}"
            out_img_path = os.path.join(img_out_dir, img_name)
            img = cv2.resize(img, dsize=(resize[0], resize[1]))
            cv2.imwrite(out_img_path, img)
            label_path = os.path.join(label_out_dir, img_name.replace(".jpg", ".txt"))
            with open(label_path, "w") as f:
                f.write(label)
    return cnt


def bbox2cpwhn(bbox, w, h):
    cpwhn = [round((bbox["x"] + bbox["width"] / 2) / w, 6),
             round((bbox["y"] + bbox["height"] / 2) / h, 6),
             round(bbox["width"] / w, 6),
             round(bbox["height"] / h, 6)]
    return cpwhn


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/지하철 역사 내 CCTV 이상행동 영상"
    parser.add_argument("--root", type=str, default=root)

    out_dir = "/media/daton/Data/datasets/aihub/subway_cctv"
    parser.add_argument("--out-dir", type=str, default=out_dir)

    prefix = "subway_cctv"
    parser.add_argument("--prefix", type=str, default=prefix)
    parser.add_argument("--execute", type=bool, default=True)
    parser.add_argument("--resize", type=int, default=[1280, 720])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
