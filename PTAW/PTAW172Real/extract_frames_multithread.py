# Extract frames by interval from input video
import argparse
import os
import time
import multiprocessing as mp
from pathlib import Path

import cv2


def extract_frames(vid_item, verbose=False):
    vid_name, vid_dir, interval, target_size, save_dir, save = vid_item
    vid_path = os.path.join(vid_dir, vid_name)
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize = [width, height] == target_size
    if verbose:
        print(f"\n--- Processing {vid_name}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"\twidth: {width}, height: {height}, fps: {fps:.2f}, total_frame: {total_frames}")
    cnt = 0
    while True:
        tmp_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, img = cap.read()
        if not ret:
            break
        if save and tmp_frame % interval == 0:
            save_path = os.path.join(save_dir, Path(vid_name).name.replace(".mp4", "") + f"_{tmp_frame}.png")
            if resize:
                img = cv2.resize(img, dsize=target_size)
            cv2.imwrite(save_path, img)
            cnt += 1
    print(f"\ttotal {cnt} frames are saved from {Path(vid_name).name}!")


def main(args):
    target_vids = args.target_videos
    save_dir = args.save_dir
    target_size = args.target_size
    num_workers = args.num_workers
    interval = args.interval
    save = args.save

    if os.path.isfile(target_vids):
        vid_names = [target_vids]
    elif os.path.isdir(target_vids):
        vid_names = [x for x in os.listdir(target_vids) if os.path.join(target_vids, x) and x.endswith(".mp4")]
    else:
        print(f"Given target videos '{target_vids}' is wrong path!")
        return
    assert len(vid_names) >= 1, f"Can't find target videos!"

    if not os.path.isdir(save_dir) and save:
        os.makedirs(save_dir)

    ts = time.time()
    pool = mp.Pool(num_workers)
    pool.map(
        extract_frames,
        zip(vid_names,
            len(vid_names) * [target_vids],
            len(vid_names) * [interval],
            len(vid_names) * [target_size],
            len(vid_names) * [save_dir],
            len(vid_names) * [save])
    )
    pool.close()
    pool.join()
    te = time.time()
    print(f"\n--- Elapsed time: {te - ts:.2f}")


def parse_args():
    parser = argparse.ArgumentParser()

    target_vids = "/media/daton/Data/datasets/PTAW/PTAW_Datasets/PTAW172Real/extracted_videos"
    parser.add_argument("--target-videos", type=str, default=target_vids)

    save_dir = "/media/daton/Data/datasets/PTAW/PTAW_Datasets/PTAW172Real/extracted_frames_by_interval"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    target_size = [1280, 720]
    parser.add_argument("--target-size", type=int, default=target_size)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--interval", type=int, default=70)
    parser.add_argument("--save", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
