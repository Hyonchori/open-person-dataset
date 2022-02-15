import argparse
import os
import time

import cv2
import numpy as np
from tqdm import tqdm


def main(args):
    target_dir = args.target_dir
    vid_list = args.vid_list
    save_dir = args.save_dir
    save_name = args.save_name
    interval = args.interval
    fps = args.fps
    width = args.width
    height = args.height
    save = args.save
    show = args.show

    print(len(vid_list))
    vid_list = [f"{x}.mp4" for x in vid_list if os.path.isfile(os.path.join(target_dir, f"{x}.mp4"))]
    print(len(vid_list))
    save_path = os.path.join(save_dir, f"{save_name}.mp4")
    if save:
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    else:
        vid_writer = None

    for vid_name in vid_list:
        vid_path = os.path.join(target_dir, vid_name)
        vid_cap = cv2.VideoCapture(vid_path)
        vid_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        time.sleep(0.2)
        print(f"\n--- Processing {vid_path}")
        time.sleep(0.2)
        for _ in tqdm(range(vid_frames)):
            ret, img = vid_cap.read()
            if not ret:
                break
            img = cv2.resize(img, dsize=(width, height))
            if show:
                cv2.imshow("img", img)
                cv2.waitKey(1)
            if vid_writer is not None:
                vid_writer.write(img)
        if vid_writer is not None:
            img = np.ones((height, width, 3), dtype=np.uint8)
            for _ in range(interval * fps):
                vid_writer.write(img)


def parse_args():
    parser = argparse.ArgumentParser()

    target_dir = "/home/daton/Desktop/tmp"
    parser.add_argument("--target-dir", type=str, default=target_dir)

    vid_list = ["2149408", "2149436", "2149469", "2149522"]
    vid_list = ["2150537", "2150657", "2150777", "2151099", "2151150", "2151254", "2151338"]
    vid_list = ["2153968", "2153997", "2154023", "2154067", "2154281", "2154480", "2154843", "2154948", "2155461",
                "2187562", "2187629", "2187716", "2191996", "2192004", "2192030", "2192058", "2192086", "2192188"]
    #vid_list = ["2192178", "2192200", "2192233", "2192258", "2192300", "2192326", "2192367", "2192392", "2192447"]
    parser.add_argument("--vid-list", nargs="+", type=str, default=vid_list)

    save_dir = "/home/daton/Desktop/gs"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    save_name = "aihub_subway_cctv_3"
    parser.add_argument("--save-name", type=str, default=save_name)

    interval = 1  # n second
    parser.add_argument("--interval", type=int, default=interval)

    fps = 3
    parser.add_argument("--fps", type=int, default=fps)

    width = 1080
    parser.add_argument("--width", type=int, default=width)

    height = 720
    parser.add_argument("--height", type=int, default=height)

    save = True
    parser.add_argument("--save", action="store_true", default=save)

    show = True
    parser.add_argument("--show", action="store_true", default=show)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
