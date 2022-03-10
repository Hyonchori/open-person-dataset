import os
import json

import cv2


SPLITS = {1: "Training", 2: "Validation"}
ACTIONS = {1: "실신", 2: "환경전도", 3: "에스컬레이터 전도", 4: "계단 전도", 5: "배회"}


def view_annot(root, target_split=None, target_action=None, target_case=None, target_scene=None):
    splits = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
    if target_split is not None:
        assert target_split in SPLITS, f"Given target split '{target_split}' not in {SPLITS}"
        splits = [SPLITS[target_split]]
    for split in splits:
        split_dir_path = os.path.join(root, split)
        actions = [x for x in os.listdir(split_dir_path) if os.path.isdir(os.path.join(split_dir_path, x))]
        if target_action is not None:
            assert target_action in ACTIONS, f"Given target action '{target_action}' not in {ACTIONS}"
            actions = [ACTIONS[target_action]]
        for action in actions:
            action_dir_path = os.path.join(split_dir_path, action)
            case_dict = {int(x.split("_")[-1]): x for x in os.listdir(action_dir_path)
                         if os.path.isdir(os.path.join(action_dir_path, x)) and "원천" in x}
            if target_case is not None:
                assert target_case in case_dict, f"Given target case '{target_case}' not in {sorted(case_dict.keys())}"
                case_dict = {target_case: case_dict[target_case]}
            annot_dict = {k: v.replace("원천", "라벨") for k, v in case_dict.items()}
            for case in case_dict:
                case_dir_path = os.path.join(action_dir_path, case_dict[case])
                scenes = [x for x in os.listdir(case_dir_path) if os.path.isdir(os.path.join(case_dir_path, x))]
                if target_scene is not None:
                    assert str(target_scene) in scenes, f"Given target scene '{target_scene}' not in {sorted(scenes)}"
                    scenes = [str(target_scene)]
                for scene in scenes:
                    scene_dir_path = os.path.join(case_dir_path, scene)
                    annot_path = os.path.join(action_dir_path, annot_dict[case], f"annotation_{scene}.json")
                    with open(annot_path) as f:
                        annot = json.load(f)
                    visualize_one_vid(scene_dir_path, annot)


def visualize_one_vid(img_dir_path, annots, txt_org=(30, 70), font_size=5, font_thickness=3, resize=(1280, 720)):
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
    label = f"{action}: {img_dir_path.split('/')[-1]}"
    imgs = sorted(os.listdir(img_dir_path))
    target_indices = [i for i, x in enumerate(annots["frames"]) if x["image"] in imgs]
    print(f"\n--- Processing {img_dir_path}")
    for img_name, target_idx in zip(imgs, target_indices):
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)
        txt_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)
        cv2.rectangle(img, (txt_org[0], txt_org[1] + 10),
                      (txt_org[0] + txt_size[0], txt_org[1] - txt_size[1] - 10), [0, 0, 0], -1)
        cv2.putText(img, label, txt_org, cv2.FONT_HERSHEY_PLAIN, font_size, [255, 255, 255], font_thickness,
                    cv2.LINE_AA)
        tmp_annots = annots["frames"][target_idx]["annotations"]
        for tmp_annot in tmp_annots:
            xyxy = bbox2xyxy(tmp_annot["label"])
            category = tmp_annot["category"]["code"]
            color = [0, 255, 255] if category == "person" else [0, 0, 255]
            img = cv2.rectangle(img, xyxy[:2], xyxy[2:], color, 2)
        img = cv2.resize(img, resize)
        cv2.imshow("img", img)
        cv2.waitKey(1)


def bbox2xyxy(bbox):
    xyxy = [bbox["x"],
            bbox["y"],
            bbox["x"] + bbox["width"],
            bbox["y"] + bbox["height"]]
    return xyxy


if __name__ == "__main__":
    root = "/media/daton/Data/datasets/지하철 역사 내 CCTV 이상행동 영상"

    target_split = 1
    target_action = 1
    target_case = None
    target_scene = None
    # actions = {1: "실신", 2: "환경전도", 3: "에스컬레이터 전도", 4: "계단 전도", 5: "배회"}
    view_annot(root,
               target_split=target_split,
               target_action=target_action,
               target_case=target_case,
               target_scene=target_scene)
