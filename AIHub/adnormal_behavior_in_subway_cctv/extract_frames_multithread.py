# Extract frames by interval or frame rate
# v1: total 1934 images, with frame rate: [0.3] (26s)
import argparse
import os
import re
import glob
import time
import multiprocessing as mp
from pathlib import Path

import cv2


SPLITS = {1: "Training", 2: "Validation"}
ACTIONS = {1: "실신", 2: "환경전도", 3: "에스컬레이터 전도", 4: "계단 전도"}


def main(args):
    root = args.root
    save_dir = args.save_dir
    run_name = args.run_name
    interval = args.interval
    frame_rate = args.frame_rate
    num_workers = args.num_workers
    target_split = args.target_split
    target_action = args.target_action
    target_num = args.target_num
    target_scene = args.target_scene
    target_size = args.target_size
    save = args.save

    assert not (interval is not None and frame_rate is not None), \
        f"At least one argument in interval, frame_rate should be None"

    save_dir = increment_path(Path(save_dir) / run_name, exist_ok=False)
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)

    ts = time.time()
    total_paths = []

    splits = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)) and x in SPLITS.values()]
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
                    assert all(x in scenes for x in target_scene), \
                        f"Some elements of '{target_scene}' are not in {scenes}"
                    scenes = target_scene

                for scene in scenes:
                    scene_dir_path = os.path.join(num_dir_path, scene)
                    total_paths.append(scene_dir_path)

    pool = mp.Pool(num_workers)
    pool.map(
        extract_frames,
        zip(total_paths,
            len(total_paths) * [save_dir],
            len(total_paths) * [target_size],
            len(total_paths) * [interval],
            len(total_paths) * [frame_rate],
            len(total_paths) * [save])
    )
    pool.close()
    pool.join()
    te = time.time()
    total_save_cnt = len([x for x in os.listdir(save_dir) if x.endswith(".png")])
    print(f"\n--- Total save count: {total_save_cnt}, Elapsed time: {te - ts:.2f}s")


def extract_frames(input_item):
    img_dir_path, save_dir_path, resize, interval, frame_rate, save = input_item
    action_dir = img_dir_path.split('/')[-2]
    num_dir = img_dir_path.split('/')[-1]
    imgs = sorted(os.listdir(img_dir_path))
    if interval is not None and frame_rate is None:
        too_short = len(imgs) // interval < 2
        if too_short:
            target_frames = [len(imgs) // 2]
        else:
            target_frames = [x for x in range(interval, len(imgs), interval)]
    elif interval is None and frame_rate is not None:
        target_frames = [int(len(imgs) * x) for x in frame_rate]
    else:
        target_frames = []
    save_cnt = 0
    max_frame_per_vid = len(target_frames)
    for i, img_name in enumerate(imgs):
        if i not in target_frames:
            continue
        else:
            save_cnt += 1
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)
        if resize is not None:
            img = cv2.resize(img, dsize=resize)
        if save:
            save_name = f"{action_dir}_{num_dir}_{i + 1}.png"
            save_path = os.path.join(save_dir_path, save_name)
            cv2.imwrite(save_path, img)
        if save_cnt >= max_frame_per_vid:
            break
    print(f"Save {save_cnt} images from {action_dir}/{num_dir} ({len(imgs)})")


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


def plot_info(img, info, font_size=2, font_thickness=2):
    label_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] * 2), [0, 0, 0], -1)
    cv2.putText(img, info, (5, int(label_size[1] * 1.5))
                , cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/지하철 역사 내 CCTV 이상행동 영상"
    parser.add_argument("--root", type=str, default=root)

    save_dir = "/media/daton/Data/datasets/지하철 역사 내 CCTV 이상행동 영상/extracted_frames"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    run_name = "exp"
    parser.add_argument("--run-name", type=str, default=run_name)

    num_workers = 8
    parser.add_argument("--num-workers", type=int, default=num_workers)

    interval = 70
    interval = None
    parser.add_argument("--interval", type=int, default=interval)

    frame_rate = [0.3]
    parser.add_argument("--frame-rate", type=float, default=frame_rate)

    # 1: Training, 2: Validation
    target_split = [1]
    target_split = None
    parser.add_argument("--target-split", type=str, default=target_split)

    # 1: "실신", 2: "환경전도", 3: "에스컬레이터 전도", 4: "계단 전도"
    target_action = [1, 2]
    target_action = None
    parser.add_argument("--target-action", type=str, default=target_action)

    # Different by action
    target_num = [1]
    target_num = None
    parser.add_argument("--target-num", type=str, default=target_num)

    # Different by num of action
    target_scene = ["2131041"]
    target_scene = None
    parser.add_argument("--target-scene", type=str, default=target_scene)

    target_size = [1280, 720]
    # target_size = None
    parser.add_argument("--target-size", type=int, default=target_size)

    parser.add_argument("--save", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)