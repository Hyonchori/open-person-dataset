# Read youtube video list and save videos
import argparse
import os
import sys
import multiprocessing as mp
from pathlib import Path
from datetime import timedelta
from collections import deque

import cv2
import pandas


def get_youtube_stream(source):
    if "youtube.com/" in source or "youtu.be" in source:
        import pafy
        source = pafy.new(source).getbest(preftype="mp4").url
    return source


def time2timedelta(tmp_time):
    time_split = tmp_time.split(".")
    if len(time_split) == 3:
        m, s, ms = time_split
        tmp_timedelta = timedelta(minutes=m, seconds=s, milliseconds=ms)
    else:  # len(time_split) == 4:
        h, m, s, ms = time_split
        tmp_timedelta = timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)
    return tmp_timedelta


def save_youtube_video(vid_item):
    print(vid_item)




def main(args):
    real_dataset_xlsx = args.real_dataset_xlsx
    save_dir = args.save_dir
    target_size = args.target_size
    num_workers = args.num_workers
    save = args.save

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    assert os.path.isfile(real_dataset_xlsx), "Given dataset_xlsx'path is wrong!"
    xlsx = pandas.read_excel(real_dataset_xlsx)
    vid_indices = xlsx.index
    vid_nums = len(vid_indices)
    vid_links = set(xlsx["Video Link"])
    vid_titles = xlsx["Video Title"]
    vid_start_times = xlsx["Start"]
    vid_end_times = xlsx["End"]

    total_start_times = []
    total_end_times = []
    total_titles = []
    for vid_link in vid_links:
        tmp_indices = xlsx["Video Link"] == vid_link
        print(f"\n--- {list(vid_titles[tmp_indices])[0]}")

        tmp_start_times = list(vid_start_times[tmp_indices])
        tmp_end_times = list(vid_end_times[tmp_indices])
        tmp_times = [["start", x] for x in tmp_start_times] + [["end", x] for x in tmp_end_times]
        tmp_times = deque(sorted(tmp_times, key=lambda item: item[1]))
        refined_times = [tmp_times.popleft()]
        last_state, last_time = refined_times[0]
        while tmp_times:
            tmp_state, tmp_time = tmp_times.popleft()
            if last_state == "start" and tmp_state == "start":
                pass
            elif last_state == "start" and tmp_state == "end":
                refined_times.append([tmp_state, tmp_time])
                last_state = tmp_state
            elif last_state == "end" and tmp_state == "start":
                refined_times.append([tmp_state, tmp_time])
                last_state = tmp_state
            else:
                refined_times[-1][1] = tmp_time
        print(refined_times)

        total_start_times.append(vid_start_times[tmp_indices])
        total_end_times.append(vid_end_times[tmp_indices])
        total_titles.append(list(vid_titles[tmp_indices])[0])

    print(len(vid_links))
    print(len(total_start_times), len(total_end_times), len(total_titles))

    return
    pool = mp.Pool(num_workers)
    pool.map(
        save_youtube_video,
        zip(vid_links, total_start_times, total_end_times, total_titles,
            vid_nums * [save_dir],
            vid_nums * [target_size],
            vid_nums * [save])
    )
    pool.close()
    pool.join()


def parse_args():
    parser = argparse.ArgumentParser()

    real_dataset_xlsx = "/media/daton/Data/datasets/PTAW/PTAW_Datasets/PTAW172Real.xlsx"
    parser.add_argument("--real-dataset-xlsx", type=str, default=real_dataset_xlsx)

    save_dir = "/media/daton/Data/datasets/PTAW/PTAW_Datasets/PTAW172Real"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    target_size = [1280, 720]
    parser.add_argument("--target-size", type=int, default=target_size)

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
