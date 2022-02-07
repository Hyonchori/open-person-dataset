import os
import datetime

import cv2
import numpy as np
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


def view_one_vid(vid_dir_path, vid_name, annot_name, txt_org=(10, 50), font_size=3, font_thickness=3):
    print(f"\n--- Processing {vid_dir_path}/{vid_name}")
    vid_path = os.path.join(vid_dir_path, vid_name)
    cap = cv2.VideoCapture(vid_path)
    annot_path = os.path.join(vid_dir_path, annot_name)
    with open(annot_path) as f:
        annot = xmltodict.parse(f.read())["annotation"]
        header = annot["header"]
        fps = float(header["fps"])
        total_frames = float(header["frames"])
        total_time = time2hms(header["duration"])
        total_time_delta = datetime.timedelta(hours=total_time[0], minutes=total_time[1], seconds=total_time[2])

        event = annot["event"]
        start_time = time2hms(event["starttime"])
        start_time_delta = datetime.timedelta(hours=start_time[0], minutes=start_time[1], seconds=start_time[2])
        duration = time2hms(event["duration"])
        duration_delta = datetime.timedelta(hours=duration[0], minutes=duration[1], seconds=duration[2])
        start_time_rate = start_time_delta / total_time_delta
        end_time_rate = (start_time_delta + duration_delta) / total_time_delta
        start_frame = max(0, int(total_frames * start_time_rate - fps * 3))
        alarm_frame = int(total_frames * start_time_rate)
        end_frame = min(total_frames, int(total_frames * end_time_rate + fps * 3))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, img = cap.read()
        img = cv2.resize(img, dsize=(WIDTH, HEIGHT))
        txt_size, _ = cv2.getTextSize(vid_name, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)
        cv2.rectangle(img, (txt_org[0], txt_org[1] + 10),
                      (txt_org[0] + txt_size[0], txt_org[1] - txt_size[1] - 10), [0, 0, 0], -1)
        cv2.putText(img, vid_name, txt_org, cv2.FONT_HERSHEY_PLAIN, font_size, [255, 255, 255], font_thickness,
                    cv2.LINE_AA)
        ref_img = np.zeros_like(img)
        if pos_frame < alarm_frame:
            ref_img[..., 1:] = 225
        else:
            ref_img[..., -1] = 225
        img = cv2.addWeighted(img, 1, ref_img, 0.5, 0)
        cv2.imshow('img', img)
        cv2.waitKey(1)
        if not ret:
            break
        if pos_frame >= end_frame:
            break


def time2hms(t):
    h, m, s = [float(x) for x in t.split(":")]
    return h, m, s


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