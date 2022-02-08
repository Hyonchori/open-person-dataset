import argparse
import os
import glob
import datetime
import re
import time
from pathlib import Path

import cv2
import xmltodict
from tqdm import tqdm


ACTIONS = {5: "05.실신(swoon)"}
WIDTH = 1280
HEIGHT = 720


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


def main(args):
    root = args.root
    max_num_per_case = args.max_num_per_case
    save_interval = args.save_interval
    execute = args.execute
    out_dir = args.out_dir
    out_name = args.out_name

    out_dir = out_dir if out_dir is not None else root
    save_dir = increment_path(Path(out_dir) / out_name, exist_ok=False)
    if execute:
        save_dir.mkdir(parents=True, exist_ok=True)

    cnt = 0
    actions = [x for x in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, x)) and x in ACTIONS.values()]
    for action in actions:
        action_dir_path = os.path.join(root, action)
        split_dict = {}
        for case in os.listdir(action_dir_path):
            split = "_".join(case.split("_")[:-1])
            if split in split_dict:
                split_dict[split].append(case)
            else:
                split_dict[split] = [case]
        for split in split_dict:
            case_dict = {int(x.split("_")[-1]): x for x in split_dict[split]}
            for case in case_dict:
                case_dir_path = os.path.join(action_dir_path, case_dict[case])
                takes = [x for x in os.listdir(case_dir_path) if os.path.isdir(os.path.join(case_dir_path, x))]
                for take in takes:
                    time.sleep(0.5)
                    print(f"\n--- Processing {case_dir_path}/{take}")
                    time.sleep(0.5)
                    take_dir_path = os.path.join(case_dir_path, take)
                    scenes = [x for x in os.listdir(take_dir_path) if x.endswith(".mp4")]
                    annots = [x.replace(".mp4", ".xml") for x in scenes]
                    for scene, annot in tqdm(zip(scenes, annots)):
                        cnt += save_only_target(take_dir_path, scene, annot, max_num_per_case, save_interval, execute, save_dir)
    print(f"Final summary: {cnt} images are saved!")


def save_only_target(vid_dir_path, vid_name, annot_name, max_num_per_case, save_interval, execute, save_dir):
    vid_path = os.path.join(vid_dir_path, vid_name)
    cap = cv2.VideoCapture(vid_path)
    annot_path = os.path.join(vid_dir_path, annot_name)
    if not os.path.isfile(annot_path):
        return 0
    with open(annot_path) as f:
        annot = xmltodict.parse(f.read())["annotation"]
        header = annot["header"]
        fps = float(header["fps"])
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_seconds = total_frames / fps
        seconds = total_seconds % 60
        minutes = total_seconds // 60
        total_time_delta = datetime.timedelta(hours=0, minutes=minutes, seconds=seconds)

        event = annot["event"]
        start_time = time2hms(event["starttime"])
        start_time_delta = datetime.timedelta(hours=start_time[0], minutes=start_time[1], seconds=start_time[2])
        duration = time2hms(event["duration"])
        duration_delta = datetime.timedelta(hours=duration[0], minutes=duration[1], seconds=duration[2])
        start_time_rate = start_time_delta / total_time_delta
        end_time_rate = (start_time_delta + duration_delta) / total_time_delta
        start_frame = int(total_frames * start_time_rate)
        end_frame = int(total_frames * end_time_rate)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tmp_frame = 0
    recent_saved_frame = 0
    cnt = 0
    while True:
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, img = cap.read()
        if not ret or pos_frame >= end_frame:
            break
        try:
            imc = cv2.resize(img, dsize=(WIDTH, HEIGHT))
        except Exception as e:
            print(str(e))
            return 0
        if pos_frame == tmp_frame:
            continue
        else:
            tmp_frame = pos_frame
            if recent_saved_frame == 0 or pos_frame - recent_saved_frame >= save_interval:
                save_path = os.path.join(save_dir, f"{vid_name}_{cnt}.png")
                if execute:
                    cv2.imwrite(save_path, imc)
                cnt += 1
                recent_saved_frame = pos_frame
                if cnt >= max_num_per_case:
                    break
    return cnt


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

    out_dir = "/media/daton/Data/datasets/aihub/adnormal_behavior_images"
    parser.add_argument("--out-dir", type=str, default=out_dir)

    parser.add_argument("--out-name", type=str, default="exp")
    parser.add_argument("--max-num-per-case", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=150)
    parser.add_argument("--execute", type=bool, default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
