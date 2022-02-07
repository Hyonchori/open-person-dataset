import os

import cv2
import xmltodict


ACTIONS = {5: "05.실신(swoon)"}
SPLITS = {1: "inside_croki", 2: "insidedoor", 3: "outsidedoor"}
WIDTH = 1280
HEIGHT = 720


def view_annot(root, target_action=None, target_split=None, target_case=None, target_take=None, target_scene=None):
    actions = [x for x in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, x))]
    if target_action is not None:
        assert target_action in ACTIONS, f"Given target action '{target_action}' not in {ACTIONS}"
        actions = [ACTIONS[target_action]]
    for action in actions:
        action_dir_path = os.path.join(root, action)
        split_dict = {}
        for case in os.listdir(action_dir_path):
            split = "_".join(case.split("_")[:-1])
            if split not in SPLITS.values():
                continue
            if split in split_dict:
                split_dict[split].append(case)
            else:
                split_dict[split] = [case]
        if target_split is not None:
            assert target_split in SPLITS, f"Given target split '{target_split}' not in {SPLITS}"
            split_dict = {target_split: split_dict[SPLITS[target_split]]}
        for split in split_dict:
            case_dict = {int(x.split("_")[-1]): x for x in split_dict[split]}
            if target_case is not None:
                assert target_case in case_dict, f"Given target case '{target_case}' not in {case_dict}"
                case_dict = {target_case: case_dict[target_case]}
            for case in case_dict:
                case_dir_path = os.path.join(action_dir_path, case_dict[case])
                takes = [x for x in os.listdir(case_dir_path) if os.path.isdir(os.path.join(case_dir_path, x))]
                if target_take is not None:
                    assert 1 <= target_take <= len(takes), f"Given target take should be between {0} and {len(takes)}"
                    takes = [sorted(takes)[target_take - 1]]
                for take in takes:
                    take_dir_path = os.path.join(case_dir_path, take)
                    scenes = [x for x in os.listdir(take_dir_path) if x.endswith(".mp4")]
                    annots = [x.replace(".mp4", ".xml") for x in scenes]
                    for scene, annot in zip(scenes, annots):
                        view_one_vid(take_dir_path, scene, annot)


def view_one_vid(vid_dir_path, vid_name, annot_name, txt_org=(30, 50), font_size=3, font_thickness=3):
    vid_path = os.path.join(vid_dir_path, vid_name)
    cap = cv2.VideoCapture(vid_path)
    annot_path = os.path.join(vid_dir_path, annot_name)
    with open(annot_path) as f:
        annot = xmltodict.parse(f.read())["annotation"]

    while True:
        ret, img = cap.read()
        img = cv2.resize(img, dsize=(WIDTH, HEIGHT))
        cv2.imshow('img', img)
        cv2.waitKey(1)
        if not ret:
            break


if __name__ == "__main__":
    root = "/media/daton/Data/datasets/이상행동 CCTV 영상"

    target_action = 5
    target_split = None
    target_case = None
    target_take = 1
    target_scene = 1
    # actions = {5: "05.실신(swoon)"}
    # splits = {1: "inside_croki", 2: "insidedoor", 3: "outsidedoor"}
    view_annot(root,
               target_action=target_action,
               target_split=target_split,
               target_case=target_case,
               target_take=target_take,
               target_scene=target_scene)