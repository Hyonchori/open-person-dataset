# Extract frames by interval from input video
import argparse
import os
import re
import time
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np


def extract_frames(vid_item, verbose=False):
    vid_name, vid_dir, interval, target_size, save_dir, save = vid_item
    vid_path = os.path.join(vid_dir, vid_name)
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize = [width, height] != target_size
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
            save_name = " ".join(Path(vid_name).name.split(".")[:-1]) + f"_{tmp_frame}.png"
            save_name = clean_str(save_name)
            save_path = os.path.join(save_dir, save_name)
            if resize:
                img = letterbox(img, target_size[::-1], auto=False)[0]
            cv2.imwrite(save_path, img)
            cnt += 1
    print(f"\ttotal {cnt} frames are saved from {Path(vid_name).name}!")


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


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

    save_dir = "/media/daton/Data/datasets/PTAW/PTAW_Datasets/PTAW172Real/extracted_frames_by_interval2"
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
