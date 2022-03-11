# Extract frames by frame rate
import argparse
import os
import re
import glob
import time
import multiprocessing as mp
from pathlib import Path

import cv2

ACTIONS = {6: "image_action_6"}


def main(args):
    root = args.root
    save_dir = args.save_dir
    run_name = args.run_name
    num_workers = args.num_workers
    frame_rate = args.frame_rate
    target_action = args.target_action
    target_action_num = args.target_action_num
    target_scene = args.target_scene
    target_cam = args.target_cam
    target_size = args.target_size
    save = args.save

    save_dir = increment_path(Path(save_dir) / run_name, exist_ok=False)
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    total_paths = []

    img_dir_path = os.path.join(root, "이미지")
    actions = [x for x in os.listdir(img_dir_path)
               if os.path.isdir(os.path.join(img_dir_path, x)) and x in ACTIONS.values()]
    if target_action is not None:
        assert all(x in ACTIONS for x in target_action), f"Some elements of '{target_action}' are not in {ACTIONS}"
        assert all(ACTIONS[x] in actions for x in target_action), \
            f"Some elements of '{[ACTIONS[x] in actions for x in target_action]}' are not in {actions}"
        actions = [ACTIONS[x] for x in target_action]

    for action in actions:
        action_dir_path = os.path.join(img_dir_path, action)
        action_nums_dict = {int(x.split("-")[-1]): x for x in os.listdir(action_dir_path)
                            if os.path.isdir(os.path.join(action_dir_path, x))}
        if target_action_num is not None:
            assert all(x in action_nums_dict for x in target_action_num), \
                f"Some elements of '{target_action_num}' not in {action_nums_dict}"
            action_nums_dict = {x: action_nums_dict[x] for x in target_action_num}

        for action_num in action_nums_dict.values():
            action_num_dir_path = os.path.join(action_dir_path, action_num)
            while len(os.listdir(action_num_dir_path)) == 1:
                action_num_dir_path = os.path.join(action_num_dir_path, os.listdir(action_num_dir_path)[0])
            scenes_dict = {}
            for scene in os.listdir(action_num_dir_path):
                scene_num = int(scene.split("_")[-1].split("-")[0])
                if scene_num in scenes_dict:
                    scenes_dict[scene_num].append(scene)
                else:
                    scenes_dict[scene_num] = [scene]
            if target_scene is not None:
                assert all(x in scenes_dict for x in target_scene), \
                    f"Some elements of '{target_scene}' not in {scenes_dict}"
                scenes_dict = {x: scenes_dict[x] for x in target_scene}

            for scene in scenes_dict.values():
                cam_dict = {int(x.split("-")[-1][1:]): x for x in scene}
                if target_cam is not None:
                    assert all(x in cam_dict for x in target_cam), \
                        f"Some elements of '{target_cam}' not in {cam_dict}"
                    cam_dict = {x: cam_dict[x] for x in target_cam}

                for cam in cam_dict.values():
                    cam_dir_path = os.path.join(action_num_dir_path, cam)
                    total_paths.append(cam_dir_path)
    pool = mp.Pool(num_workers)
    pool.map(
        extract_frames,
        zip(total_paths,
            len(total_paths) * [save_dir],
            len(total_paths) * [target_size],
            len(total_paths) * [frame_rate],
            len(total_paths) * [save])
    )
    pool.close()
    pool.join()
    te = time.time()
    total_save_cnt = len([x for x in os.listdir(save_dir) if x.endswith(".png")])
    print(f"\n--- Total save count: {total_save_cnt}, frame_rate: {frame_rate}, Elapsed time: {te - ts:.2f}s")


def extract_frames(input_item):
    img_dir_path, save_dir_path, resize, frame_rate, save = input_item
    cam_name = Path(img_dir_path).name
    imgs = sorted(os.listdir(img_dir_path))
    if frame_rate is not None:
        target_frames = [int(len(imgs) * x) for x in frame_rate]
    else:
        target_frames = list(range(len(imgs)))
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
            save_name = img_name.replace(".jpg", ".png")
            save_path = os.path.join(save_dir_path, save_name)
            cv2.imwrite(save_path, img)
        if save_cnt >= max_frame_per_vid:
            break
    print(f"Save {save_cnt} images from {cam_name} ({len(imgs)})")


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


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/사람동작 영상"
    parser.add_argument("--root", type=str, default=root)

    save_dir = "/media/daton/Data/datasets/사람동작 영상/extracted_frames"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    run_name = "exp"
    parser.add_argument("--run-name", type=str, default=run_name)

    num_workers = 4
    parser.add_argument("--num-workers", type=int, default=num_workers)

    frame_rate = [0.8]
    parser.add_argument("--frame-rate", type=float, default=frame_rate)

    # 6: fall-down
    target_action = [6]
    target_action = None
    parser.add_argument("--target-action", type=str, default=target_action)

    # Different by action
    target_action_num = [3]
    target_action_num = None
    parser.add_argument("--target-action-num", type=str, default=target_action_num)

    # Different by action
    target_scene = [3]
    target_scene = None
    parser.add_argument("--target-scene", type=str, default=target_scene)

    # Different by scene
    target_cam = [3]
    target_cam = None
    parser.add_argument("--target-cam", type=str, default=target_cam)

    target_size = [1280, 720]
    parser.add_argument("--target-size", type=int, default=target_size)

    parser.add_argument("--save", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
