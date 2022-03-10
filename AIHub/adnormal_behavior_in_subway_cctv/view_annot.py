# View annotations of '지하철 역사 내 CCTV 이상행동 영상' dataset
import argparse
import os
import json

import cv2


SPLITS = {1: "Training", 2: "Validation"}
ACTIONS = {1: "실신", 2: "환경전도", 3: "에스컬레이터 전도", 4: "계단 전도"}


def main(args):
    root = args.root
    target_split = args.target_split
    target_action = args.target_action
    target_num = args.target_num
    target_scene = args.target_scene
    view = args.view
    view_size = args.view_size

    splits = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
    if target_split is not None:
        assert all(SPLITS[x] in splits for x in target_split), \
            f"Some elements of '{[SPLITS[x] for x in target_split]}' are not in {splits}"
        splits = [SPLITS[x] for x in target_split]

    for split in splits:
        split_dir_path = os.path.join(root, split)
        actions = [x for x in os.listdir(split_dir_path) if os.path.isdir(os.path.join(split_dir_path, x))]
        if target_action is not None:
            assert all(ACTIONS[x] in actions for x in target_action), \
                f"Some elements of '{[ACTIONS[x] for x in target_action]}' are not in {actions}"
            actions = [ACTIONS[x] for x in target_action]

        for action in actions:
            action_dir_path = os.path.join(split_dir_path, action)
            nums_dict = {int(x.split("_")[-1]): x for x in os.listdir(action_dir_path)
                         if os.path.isdir(os.path.join(action_dir_path, x)) and "원천" in x}
            if target_num is not None:
                assert all(x in nums_dict for x in target_num), \
                    f"Some elements of '{target_num}' are not in {nums_dict.keys()}"
                nums_dict = {x: nums_dict[x] for x in target_num}

            for num in nums_dict.values():
                num_dir_path = os.path.join(action_dir_path, num)
                scenes = [x for x in os.listdir(num_dir_path) if os.path.isdir(os.path.join(num_dir_path, x))]
                if target_scene is not None:
                    assert all(x in scenes for x in target_scene),  \
                        f"Some elements of '{target_scene}' are not in {scenes}"
                    scenes = target_scene

                for scene in scenes:
                    scene_dir_path = os.path.join(num_dir_path, scene)
                    annot_path = os.path.join(action_dir_path, num.replace("원천", "라벨"), f"annotation_{scene}.json")
                    if view:
                        visualize_one_vid(scene_dir_path, annot_path, resize=view_size)


def visualize_one_vid(img_dir_path, annot_path, resize=(1280, 720)):
    action_dir = img_dir_path.split('/')[-2]
    if "환경" in action_dir:
        action = "Fall down in env"
    elif "계단" in action_dir:
        action = "Fall down in stair"
    elif "에스컬레이터" in action_dir:
        action = "Fall down in escalator"
    elif "배회" in action_dir:
        action = "Loitering"
    else:
        action = "Fall down"
    info = f"{action}: {img_dir_path.split('/')[-1]}"
    imgs = sorted(os.listdir(img_dir_path))
    with open(annot_path) as f:
        annots = json.load(f)
    target_indices = [i for i, x in enumerate(annots["frames"]) if x["image"] in imgs]
    print(f"\n--- Processing {img_dir_path}")
    for img_name, target_idx in zip(imgs, target_indices):
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)
        plot_info(img, info)
        tmp_annots = annots["frames"][target_idx]["annotations"]
        for tmp_annot in tmp_annots:
            xyxy = bbox2xyxy(tmp_annot["label"])
            category = tmp_annot["category"]["code"]
            color = [0, 255, 255] if category == "person" else [0, 0, 255]
            img = cv2.rectangle(img, xyxy[:2], xyxy[2:], color, 2)
            plot_label(img, xyxy, category, color)
        if resize is not None:
            img = cv2.resize(img, dsize=resize)
        cv2.imshow("img", img)
        cv2.waitKey(1)


def bbox2xyxy(bbox):
    xyxy = [bbox["x"],
            bbox["y"],
            bbox["x"] + bbox["width"],
            bbox["y"] + bbox["height"]]
    return xyxy


def plot_info(img, info, font_size=3, font_thickness=3):
    label_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] * 2), [0, 0, 0], -1)
    cv2.putText(img, info, (5, int(label_size[1] * 1.5))
                , cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)


def plot_label(img, xyxy, label, color, font_size=3, font_thickness=3):
    txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
    txt_bk_color = [int(c * 0.7) for c in color]
    cv2.rectangle(img, xyxy[:2], (xyxy[0] + txt_size[0] + 1, xyxy[1] + int(txt_size[1] * 1.5)),
                  txt_bk_color, -1)
    cv2.putText(img, label, (xyxy[0], xyxy[1] + int(txt_size[1] * 1.2)),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/지하철 역사 내 CCTV 이상행동 영상"
    parser.add_argument("--root", type=str, default=root)

    # 1: Training, 2: Validation
    target_split = [1]
    # split = None
    parser.add_argument("--target-split", type=str, default=target_split)

    # 1: "실신", 2: "환경전도", 3: "에스컬레이터 전도", 4: "계단 전도"
    target_action = [1, 2]
    # target_action = None
    parser.add_argument("--target-action", type=str, default=target_action)

    # Different by action
    target_num = [1]
    target_num = None
    parser.add_argument("--target-num", type=str, default=target_num)

    # Different by num of action
    target_scene = ["2131041"]
    target_scene = None
    parser.add_argument("--target-scene", type=str, default=target_scene)

    parser.add_argument("--view", action="store_true", default=True)
    parser.add_argument("--view-size", action="store_true", default=[1280, 720])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
