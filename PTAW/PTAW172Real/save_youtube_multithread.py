# Read youtube video list and save videos
import argparse
import os
import multiprocessing as mp
from datetime import timedelta
from collections import deque

import cv2
import pandas


EXCEPTION_LIST = [
    "https://www.youtube.com/watch?v=BQXYQTpelSA",
    "https://www.youtube.com/watch?v=ijxATCTNCZc"
]


def get_youtube_stream(source):
    if "youtube.com/" in source or "youtu.be" in source:
        import pafy
        source = pafy.new(source).getbest(preftype="mp4").url
    return source


def time2timedelta(tmp_time):
    time_split = list(map(int, tmp_time.split(".")))
    if len(time_split) == 3:
        m, s, ms = time_split
        tmp_timedelta = timedelta(minutes=m, seconds=s, milliseconds=ms)
    else:  # len(time_split) == 4:
        h, m, s, ms = time_split
        tmp_timedelta = timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)
    return tmp_timedelta


def save_youtube_video(vid_item):
    vid_link, vid_start_times, vid_end_times, vid_title, save_dir, target_size, save = vid_item
    if vid_link in EXCEPTION_LIST:
        return
    print(f"\n--- Start saving video from {vid_link}, {vid_title}")
    vid_source = get_youtube_stream(vid_link)
    cap = cv2.VideoCapture(vid_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30 if fps == 0 else fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = int(total_frames / fps)
    total_timedelta = timedelta(seconds=total_seconds)
    print(f"\tfps: {fps}, total_times: {total_timedelta}, total_frames: {total_frames}, total_interval: {len(vid_start_times)}")

    for i, (start_time, end_time) in enumerate(zip(vid_start_times, vid_end_times)):
        if save:
            save_path = os.path.join(save_dir, vid_title + f"_{i + 1}.mp4")
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, target_size)
        start_time_delta = time2timedelta(start_time)
        start_time_rate = start_time_delta / total_timedelta
        start_frame = int(total_frames * start_time_rate)
        end_time_delta = time2timedelta(end_time)
        end_time_rate = end_time_delta / total_timedelta
        end_frame = int(total_frames * end_time_rate)
        print(f"\t\t{i + 1}'s tmp_start: {start_time_delta}, tmp_end: {end_time_delta}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        tmp_frame1 = start_frame
        while True:
            ret, img = cap.read()
            if not ret:
                break
            tmp_frame2 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if tmp_frame1 == tmp_frame2:
                continue
            else:
                tmp_frame1 = tmp_frame2
            img = cv2.resize(img, dsize=target_size)
            if save:
                vid_writer.write(img)
            if tmp_frame1 == end_frame:
                break
    cap.release()
    print(f"\tSave {i + 1} videos from {vid_link}")


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
    vid_links = sorted(list(set(xlsx["Video Link"])))
    vid_titles = xlsx["Video Title"]
    vid_start_times = xlsx["Start"]
    vid_end_times = xlsx["End"]

    total_start_times = []
    total_end_times = []
    total_titles = []
    for vid_link in vid_links:
        tmp_indices = xlsx["Video Link"] == vid_link
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
        total_start_times.append([x[1] for x in refined_times if x[0] == "start"])
        total_end_times.append([x[1] for x in refined_times if x[0] == "end"])
        total_titles.append(list(vid_titles[tmp_indices])[0])

    #for vid_link, start_times, end_times, title in zip(vid_links, total_start_times, total_end_times, total_titles):
    #    save_youtube_video((vid_link, start_times, end_times, title, save_dir, target_size, save))
    #return
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

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
