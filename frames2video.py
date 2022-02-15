import argparse
import os
import time
from pathlib import Path

import cv2
from tqdm import tqdm


def main(args):
    source = args.source
    expansion = args.expansion
    save_dir = args.save_dir
    save_name = args.save_name
    fps = args.fps
    width = args.width
    height = args.height
    show = args.show
    save = args.save

    imgs = sorted([x for x in os.listdir(source) if x.endswith(f".{expansion}")])
    save_name = save_name if save_name is not None else Path(source).name
    save_path = os.path.join(save_dir, f"{save_name}.mp4")
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print(f"\n---Processing {source}")
    time.sleep(0.5)
    for img_name in tqdm(imgs):
        img_path = os.path.join(source, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=(width, height))

        if show:
            cv2.imshow("img", img)
            freq = 1 / fps
            wait = int(freq * 1e3)
            cv2.waitKey(wait)

        if save:
            vid_writer.write(img)


def parse_args():
    parser = argparse.ArgumentParser()

    source = "/media/daton/Data/datasets/지하철 cctv/지하철 역사 내 CCTV 이상행동 영상/Training/배회/[원천]배회_4/2119646"
    parser.add_argument("--source", type=str, default=source)

    # expansion list: ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    expansion = "jpg"
    parser.add_argument("--expansion", type=str, default=expansion)

    save_dir = "/home/daton/Desktop/tmp"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    save_name = None
    parser.add_argument("--save-name", type=str, default=save_name)

    fps = 3
    parser.add_argument("--fps", type=int, default=fps)

    width = 1080
    parser.add_argument("--width", type=int, default=width)

    height = 720
    parser.add_argument("--height", type=int, default=height)

    show = False
    parser.add_argument("--show", action="store_true", default=show)

    save = True
    parser.add_argument("--save", action="store_true", default=save)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
