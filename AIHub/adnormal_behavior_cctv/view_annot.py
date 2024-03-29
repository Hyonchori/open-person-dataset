# View annotations of '이상행동 CCTV 영상' dataset
import argparse
import os
from datetime import timedelta

import cv2
import xmltodict
import numpy as np


ACTIONS = {5: "05.실신(swoon)", 7: "07.침입(trespass)"}
SPLITS = {1: "inside_croki", 2: "insidedoor", 3: "outsidedoor"}


def main(args):
    root = args.root
    only_view_event = args.only_view_event
    start_margin = args.start_margin
    end_margin = args.end_margin
    view_size = args.view_size
    target_action = args.target_action
    target_split = args.target_split
    target_split_num = args.target_split_num
    target_scene = args.target_scene
    target_cam_idx = args.target_cam_idx

    actions = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)) and x in ACTIONS.values()]
    if target_action is not None:
        assert all(x in ACTIONS for x in target_action), \
            f"Some elements of '{target_action}' are not in {ACTIONS}"
        assert all(ACTIONS[x] in actions for x in target_action), \
            f"Some elements of '{[ACTIONS[x] for x in target_action]}' are not in {actions}"
        actions = [ACTIONS[x] for x in target_action]

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
            assert all(x in SPLITS for x in target_split), \
                f"Some elements of '{target_split}' are not in {SPLITS}"
            assert all(SPLITS[x] in split_dict for x in target_split), \
                f"Some elements of '{[SPLITS[x] for x in target_split]}' are not in {split_dict}"
            split_dict = {SPLITS[x]: split_dict[SPLITS[x]] for x in target_split}

        for splits in split_dict.values():
            if target_split_num is not None:
                splits = [x for x in sorted(splits) if int(x.split("_")[-1]) in target_split_num]
                assert len(splits) > 0, f"Given numbers '{target_split_num}' are not valid in {splits}"

            for tmp_split in splits:
                split_dir_path = os.path.join(action_dir_path, tmp_split)
                scenes = [x for x in sorted(os.listdir(split_dir_path))
                          if os.path.isdir(os.path.join(split_dir_path, x))]
                if target_scene is not None:
                    scenes = [x for x in scenes if x in target_scene]
                    assert len(scenes) > 0, f"Given scenes '{target_scene}' are not valid in {scenes}"

                for scene in scenes:
                    scene_dir_path = os.path.join(split_dir_path, scene)
                    cams = [x for x in sorted(os.listdir(scene_dir_path))
                            if os.path.isfile(os.path.join(scene_dir_path, x)) and x.endswith(".mp4")]
                    if target_cam_idx is not None:
                        cams = [x for i, x in enumerate(cams) if i + 1 in target_cam_idx]
                        assert len(cams) > 0, f"Given cams '{target_cam_idx}' are not valid in {cams}"

                    for cam in cams:
                        vid_path = os.path.join(scene_dir_path, cam)
                        annot_path = vid_path.replace(".mp4", ".xml")
                        if not os.path.isfile(annot_path):
                            continue
                        visualize_one_vid(vid_path, annot_path, only_view_event, start_margin, end_margin, view_size)


def visualize_one_vid(vid_path, annot_path, only_view_event, start_margin, end_margin, view_size, fs=2, ft=2):
    cap = cv2.VideoCapture(vid_path)
    with open(annot_path) as f:
        annot = xmltodict.parse(f.read())["annotation"]
        vid_name = annot["filename"]
        header = annot["header"]
        fps = float(header["fps"])
        width = int(annot["size"]["width"])
        height = int(annot["size"]["height"])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = total_frames / fps
        total_time_delta = timedelta(seconds=int(total_seconds))
        print(f"\n--- Processing {vid_name}")
        print(f"\twidth: {width}, height: {height}, fps: {fps}, total_frames: {total_frames}, duration: {total_time_delta}")

        if only_view_event:
            event = annot["event"]
            start_time = time2hms(event["starttime"])
            start_time_delta = timedelta(hours=start_time[0], minutes=start_time[1], seconds=start_time[2])
            duration = time2hms(event["duration"])
            duration_delta = timedelta(hours=duration[0], minutes=duration[1], seconds=int(duration[2]))
            start_time_rate = start_time_delta / total_time_delta
            end_time_rate = (start_time_delta + duration_delta) / total_time_delta
            start_frame = max(0, int(total_frames * start_time_rate - fps * start_margin))
            alarm_start_frame = int(total_frames * start_time_rate)
            alarm_end_frame = int(total_frames * end_time_rate)
            end_frame = min(total_frames, int(total_frames * end_time_rate + fps * end_margin))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"\talarm start: {alarm_start_frame}, alarm duration: {duration_delta}")

        while True:
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, img = cap.read()
            if not ret:
                break
            ref_img = np.zeros_like(img)
            if alarm_start_frame <= pos_frame < alarm_end_frame:
                ref_img[..., -1] = 225
            img = cv2.addWeighted(img, 1, ref_img, 0.3, 0)
            if view_size is not None:
                img = cv2.resize(img, dsize=view_size)
            info = f"{vid_name}: {pos_frame} / {total_frames}"
            plot_info(img, info)
            cv2.imshow("img", img)
            cv2.waitKey(1)
            if pos_frame >= end_frame:
                break


def time2hms(t):
    if len(t.split(":")) == 3:
        h, m, s = [float(x) for x in t.split(":")]
        return h, m, s
    else:
        m, s = [float(x) for x in t.split(":")]
        return 0, m, s


def plot_info(img, info, font_size=2, font_thickness=2):
    label_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] * 2), [0, 0, 0], -1)
    cv2.putText(img, info, (5, int(label_size[1] * 1.5))
                , cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/이상행동 CCTV 영상"
    parser.add_argument("--root", type=str, default=root)

    parser.add_argument("--only-view-event", action="store_true", default=True)
    parser.add_argument("--start-margin", type=int, default=5)
    parser.add_argument("--end-margin", type=int, default=5)
    parser.add_argument("--view-size", type=int, default=[1280, 720])

    # 5: swoon, 7: trespass
    target_action = [5, 7]
    target_action = None
    parser.add_argument("--target-action", type=int, default=target_action)

    # 1: inside_croki, 2: insidedoor, 3: outsidedoor
    target_split = [3]
    #target_split = None
    parser.add_argument("--target-split", type=int, default=target_split)

    # Different by action, split
    target_split_num = [1, 2, 3, 4, 5]
    #target_split_num = None
    parser.add_argument("--target-split-num", type=int, default=target_split_num)

    # Different by action, split
    target_scene = ["146-6", "147-1"]
    target_scene = None
    parser.add_argument("--target-scene", type=str, default=target_scene)

    # Different by scene
    target_cam_idx = [6]
    target_cam_idx = None
    parser.add_argument("--target-cam-idx", type=int, default=target_cam_idx)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
