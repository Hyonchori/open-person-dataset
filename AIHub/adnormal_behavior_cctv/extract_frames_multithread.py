# Extract frames from alarm start frame to alarm end frame by interval
# v1: Total save count: 1339, Interval: 200, Elapsed time: 980.39s
import argparse
import os
import re
import glob
import time
import multiprocessing as mp
from pathlib import Path
from datetime import timedelta

import cv2
import xmltodict

ACTIONS = {5: "05.실신(swoon)", 7: "07.침입(trespass)"}
SPLITS = {1: "inside_croki", 2: "insidedoor", 3: "outsidedoor"}


def main(args):
    root = args.root
    save_dir = args.save_dir
    run_name = args.run_name
    interval = args.interval
    num_workers = args.num_workers
    max_frame_per_vid = args.max_frame_per_vid
    target_size = args.target_size
    save = args.save
    target_action = args.target_action
    target_split = args.target_split
    target_split_num = args.target_split_num
    target_scene = args.target_scene
    target_cam_idx = args.target_cam_idx

    save_dir = increment_path(Path(save_dir) / run_name, exist_ok=False)
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)

    ts = time.time()
    total_paths = []

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
                        else:
                            total_paths.append(vid_path)

    pool = mp.Pool(num_workers)
    pool.map(
        extract_frames,
        zip(total_paths,
            len(total_paths) * [save_dir],
            len(total_paths) * [interval],
            len(total_paths) * [max_frame_per_vid],
            len(total_paths) * [target_size],
            len(total_paths) * [save])
    )
    te = time.time()
    total_save_cnt = len([x for x in os.listdir(save_dir) if x.endswith(".png")])
    print(f"\n--- Total save count: {total_save_cnt}, Interval: {interval}, Elapsed time: {te - ts:.2f}s")


def extract_frames(input_item):
    vid_path, save_dir, interval, max_frame_per_vid, target_size, save = input_item
    annot_path = vid_path.replace(".mp4", ".xml")
    cap = cv2.VideoCapture(vid_path)
    with open(annot_path) as f:
        annot = xmltodict.parse(f.read())["annotation"]
        vid_name = annot["filename"]
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = total_frames / fps
        total_time_delta = timedelta(seconds=int(total_seconds))

        event = annot["event"]
        start_time = time2hms(event["starttime"])
        start_time_delta = timedelta(hours=start_time[0], minutes=start_time[1], seconds=start_time[2])
        duration = time2hms(event["duration"])
        duration_delta = timedelta(hours=duration[0], minutes=duration[1], seconds=int(duration[2]))
        start_time_rate = start_time_delta / total_time_delta
        end_time_rate = (start_time_delta + duration_delta) / total_time_delta
        alarm_start_frame = int(total_frames * start_time_rate)
        alarm_end_frame = int(total_frames * end_time_rate)
        cap.set(cv2.CAP_PROP_POS_FRAMES, alarm_start_frame)

    save_cnt = 0
    target_length = alarm_end_frame - alarm_start_frame
    too_short = target_length // interval < 1
    if too_short:
        target_frames = [target_length // 2]
    else:
        target_frames = [x for x in range(0, target_length - 1, interval)]
    for i in range(target_length):
        ret, img = cap.read()
        if not ret:
            break
        if i not in target_frames:
            continue
        else:
            save_cnt += 1
        if target_size is not None:
            img = cv2.resize(img, dsize=target_size)
        if save:
            save_name = f"{vid_name.replace('.mp4', '')}_{i}.png"
            save_path = os.path.join(save_dir, save_name)
            cv2.imwrite(save_path, img)
        if save_cnt >= max_frame_per_vid:
            break
    split_dir = vid_path.split("/")[-2]
    print(f"Save {save_cnt} images from {split_dir}/{vid_name} ({target_length})")
    cap.release()


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def time2hms(t):
    if len(t.split(":")) == 3:
        h, m, s = [float(x) for x in t.split(":")]
        return h, m, s
    else:
        m, s = [float(x) for x in t.split(":")]
        return 0, m, s


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/이상행동 CCTV 영상"
    parser.add_argument("--root", type=str, default=root)

    save_dir = "/media/daton/Data/datasets/이상행동 CCTV 영상/extracted_frames"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    run_name = "exp"
    parser.add_argument("--run-name", type=str, default=run_name)

    num_workers = 4
    parser.add_argument("--num-workers", type=int, default=num_workers)

    interval = 100
    parser.add_argument("--interval", type=int, default=interval)

    max_frame_per_vid = 3
    parser.add_argument("--max_frame_per_vid", type=int, default=max_frame_per_vid)

    parser.add_argument("--only-view-event", action="store_true", default=True)
    parser.add_argument("--start-margin", type=int, default=5)
    parser.add_argument("--end-margin", type=int, default=5)
    parser.add_argument("--target-size", type=int, default=[1280, 720])
    parser.add_argument("--save", action="store_true", default=True)

    # 5: swoon, 7: trespass
    target_action = [5, 7]
    target_action = None
    parser.add_argument("--target-action", type=int, default=target_action)

    # 1: inside_croki, 2: insidedoor, 3: outsidedoor
    target_split = [3]
    # target_split = None
    parser.add_argument("--target-split", type=int, default=target_split)

    # Different by action, split
    target_split_num = [1, 2, 3, 4, 5]
    target_split_num = None
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
