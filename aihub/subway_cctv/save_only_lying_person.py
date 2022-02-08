import argparse
import os
import json


def main(args):
    root = args.root
    max_num_per_case = args.max_num_per_case
    save_interval = args.save_interval
    box_ratio = args.box_ratio
    execute = args.execute

    save_cnt = 0
    remove_cnt = 0
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
                for scene in scenes:
                    scene_dir_path = os.path.join(case_dir_path, scene)
                    target_annot_path = os.path.join(action_dir_path, annot_dict[case], f"annotation_{scene}.json")
                    with open(target_annot_path) as f:
                        target_annot = json.load(f)
                    cnt1, cnt2 = save_only_target(
                        scene_dir_path, target_annot, max_num_per_case, save_interval, box_ratio, execute)
                    save_cnt += cnt1
                    remove_cnt += cnt2
    print(f"\nTotal saved images: {save_cnt}, Total removed images: {remove_cnt}")


def save_only_target(img_dir_path, annots, max_num_per_case, save_interval, box_ratio, execute):
    imgs = sorted(os.listdir(img_dir_path), reverse=True)
    target_indices = [i for i, x in enumerate(annots["frames"]) if x["image"] in imgs]
    cnt = 0
    tmp_num = None
    save_imgs = []
    for img_name, target_idx in zip(imgs, target_indices):
        img_num = int(img_name.split(".")[0].split("_")[-1])
        tmp_annots = annots["frames"][target_idx]["annotations"]
        for tmp_annot in tmp_annots:
            bbox = tmp_annot["label"]
            if None in bbox.values():
                continue
            width = bbox["width"]
            height = bbox["height"]
            if width == 0 or height == 0:
                continue
            ratio = height / width
            if cnt <= max_num_per_case and \
                ratio <= box_ratio and \
                    (tmp_num is None or tmp_num - img_num >= save_interval):
                save_imgs.append(img_name)
                tmp_num = img_num
                cnt += 1
                if cnt == max_num_per_case:
                    break
        if cnt == max_num_per_case:
            break
    remove_imgs = list(set(imgs) - set(save_imgs))
    print(f"\n--- {img_dir_path}")
    print(len(imgs), len(save_imgs), len(remove_imgs))
    if execute:
        print(f"\n--- Processing {img_dir_path}")
        for img_name in remove_imgs:
            img_path = os.path.join(img_dir_path, img_name)
            os.remove(img_path)
        print(f"\tdelete {len(remove_imgs)} images")
    return len(save_imgs), len(remove_imgs)


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/지하철 역사 내 CCTV 이상행동 영상"
    parser.add_argument("--root", type=str, default=root)

    parser.add_argument("--max-num-per-case", type=int, default=3)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--box-ratio", type=float, default=0.85)  # height / width
    parser.add_argument("--execute", type=bool, default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
