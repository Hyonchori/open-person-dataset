import argparse
import os
import datetime
import time

import cv2
import xmltodict
import numpy as np
from tqdm import tqdm


def main(args):
    target_dir = args.target_dir
    vid_list = args.vid_list
    label_dir_list = args.label_dir_list
    save_dir = args.save_dir
    save_name = args.save_name
    vid_interval = args.vid_interval
    event_start_interval = args.event_start_interval
    event_end_interval = args.event_end_interval
    event_duration = args.event_duration
    fps = args.fps
    width = args.width
    height = args.height
    save = args.save
    show = args.show

    tmp_vid_len = len(vid_list)
    vid_list = [f"{x}.mp4" for x in vid_list if os.path.isfile(os.path.join(target_dir, f"{x}.mp4"))]
    assert len(vid_list) == tmp_vid_len, "Something wrong video name exists!"

    label_path_dict = {}
    for label_dir in label_dir_list:
        label_dir_path = os.path.join(target_dir, label_dir)
        for label_name in os.listdir(label_dir_path):
            if (label_name.endswith(".xml")) and (label_name.replace(".xml", ".mp4") in vid_list):
                label_path_dict[label_name.replace(".xml", "")] = os.path.join(label_dir_path, label_name)

    save_path = os.path.join(save_dir, f"{save_name}.mp4")
    if save:
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    else:
        vid_writer = None

    for vid_name in vid_list:
        label_path = label_path_dict[vid_name.replace(".mp4", "")]
        label = get_gt_from_annot(label_path)
        vid_path = os.path.join(target_dir, vid_name)
        time.sleep(0.1)
        print(f"\n--- Processing {vid_path}")
        time.sleep(0.1)
        vid_cap = cv2.VideoCapture(vid_path)
        vid_fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
        vid_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(0, int(vid_frames * label["start_frame_rate"] - vid_fps * event_start_interval))
        if event_duration is None:
            end_frame = min(vid_frames, int(vid_frames * label["end_frame_rate"] + vid_fps * event_end_interval))
        else:
            end_frame = min(vid_frames, int(vid_frames * label["start_frame_rate"] + vid_fps * event_duration))
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in tqdm(range(end_frame - start_frame)):
            ret, img = vid_cap.read()
            if not ret:
                break
            img = cv2.resize(img, dsize=(width, height))
            if show:
                cv2.imshow("img", img)
                cv2.waitKey(1)
            if vid_writer is not None:
                vid_writer.write(img)
        if len(vid_list) != 1 and vid_writer is not None:
            img = np.ones((height, width, 3), dtype=np.uint8)
            for _ in range(vid_interval * vid_fps):
                vid_writer.write(img)
    print(f"\nsave {save_name}!")


def time2hms(t):
    if len(t.split(":")) == 3:
        h, m, s = [float(x) for x in t.split(":")]
        return h, m, s
    else:
        m, s = [float(x) for x in t.split(":")]
        return 0, m, s


def get_gt_from_annot(annot_path):
    gt = {}
    with open(annot_path) as f:
        annot = xmltodict.parse(f.read())
    clip = annot["KisaLibraryIndex"]["Library"]["Clip"]
    alarm = clip["Alarms"]["Alarm"]
    vid_time = time2hms(clip["Header"]["Duration"])
    start_time = time2hms(alarm["StartTime"])
    duration = time2hms(alarm["AlarmDuration"])
    alarm_description = alarm["AlarmDescription"]
    vid_time_delta = datetime.timedelta(hours=vid_time[0], minutes=vid_time[1], seconds=vid_time[2])
    start_time_delta = datetime.timedelta(hours=start_time[0], minutes=start_time[1], seconds=start_time[2])
    duration_delta = datetime.timedelta(hours=duration[0], minutes=duration[1], seconds=duration[2])
    start_frame_rate = start_time_delta / vid_time_delta
    end_frame_rate = (start_time_delta + duration_delta) / vid_time_delta
    gt["alarm_class"] = alarm_description
    gt["start_frame_rate"] = start_frame_rate
    gt["end_frame_rate"] = end_frame_rate
    return gt


def parse_args():
    parser = argparse.ArgumentParser()

    target_dir = "/home/daton/Desktop/gs/loitering"
    target_dir = "/home/daton/Desktop/gs/intrusion"
    parser.add_argument("--target-dir", type=str, default=target_dir)


    #  loitering videos from KISA dataset
    vid_list = ["C045100_001", "C045100_002", "C045100_003", "C045100_004", "C045300_003", "C045300_004"]
    vid_list = ["C001101_003", "C001201_004"]
    vid_list = ["C007101_001", "C007201_002", "C007201_005"]
    vid_list = ["C047100_006", "C047100_008"]
    vid_list = ["C049100_002", "C049100_005", "C049100_011", "C049200_005"]
    vid_list = ["C051100_001", "C051100_003", "C051100_004", "C051100_005"]
    vid_list = ["C052100_002", "C052100_005", "C052100_017", "C052300_013"]
    vid_list = ["C054100_002", "C054100_005"]
    vid_list = ["C055200_001", "C055200_005", "C055200_011"]
    vid_list = ["C056200_002", "C056200_004"]
    vid_list = ["C058100_002", "C058100_004", "C058200_008"]
    vid_list = ["C002201_004", "C004301_003", "C005101_002", "C006101_002", "C008201_002", "C008301_005", "C017101_001", "C050100_005"]
    vid_list = ["C082100_003", "C086100_005", "C087100_006", "C099100_005", "C104300_032"]

    #  intrusion videos from KISA dataset
    vid_list = ["C001202_001", "C001302_004"]  # 2
    vid_list = ["C002202_001", "C002202_003"]  # 2
    vid_list = ["C013102_002", "C013202_003"]  # 2
    vid_list = ["C016202_011", "C016302_007"]  # 2
    vid_list = ["C045101_002", "C045101_003", "C045101_004", "C045101_006", "C045101_009", "C045101_010", "C045201_011", "C045301_004", "C045301_006"]  # 9
    vid_list = ["C050101_011", "C050101_013", "C050101_014", "C050101_015", "C050201_010", "C050301_009", "C050301_010", "C050301_011"]  # 8
    vid_list = ["C055101_005", "C055101_007", "C055201_002", "C055201_005", "C055201_007", "C055301_001", "C055301_007", "C055301_010"]  # 8
    vid_list = ["C058101_002", "C058101_013"]  # 2
    vid_list = ["C058301_001", "C058301_010", "C058301_015"]  # 3
    vid_list = ["C082101_002", "C082201_005"]  # 2
    vid_list = ["C092101_001", "C092201_002"]  # 2
    vid_list = ["C106201_017", "C106301_024"]  # 2
    # 44
    vid_list = ["C003102_001"]  # 1
    vid_list = ["C019102_001"]  # 1
    vid_list = ["C021102_001"]  # 1
    vid_list = ["C090201_003"]  # 1
    #vid_list = ["C098201_004"]  # 1   ,remove this video (can't detect person well)
    vid_list = ["C110101_004"]  # 1   ,newly add this video
    vid_list = ["C104301_001"]  # 1
    vid_list = ["C114101_001"]  # 1

    parser.add_argument("--vid-list", nargs="+", type=str, default=vid_list)

    save_name = "KISA_intrusion_19"
    parser.add_argument("--save-name", type=str, default=save_name)

    label_dir_list = ["/media/daton/SAMSUNG/3. 연구개발분야/3. 바이오인식(1500개)/1. 얼굴(1410개)",
                      "/media/daton/SAMSUNG/3. 연구개발분야/3. 바이오인식(1500개)/2. 걸음걸이(90개)",
                      "/media/daton/SAMSUNG/3. 연구개발분야/1. 해외환경(1500개)/1. 배회(325개)",
                      "/media/daton/SAMSUNG/3. 연구개발분야/1. 해외환경(1500개)/2. 침입(170개)",
                      "/media/daton/SAMSUNG/4. 민간분야(2021 특수환경)/distribution"]
    parser.add_argument("--label-dir-list", nargs="+", type=str, default=label_dir_list)

    save_dir = "/home/daton/Desktop/gs/intrusion_gs_v3"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    vid_interval = 3  # n second
    parser.add_argument("--vid-interval", type=int, default=vid_interval)

    event_start_interval = 35  # n second, {loitering: 15, intrusion: 5}
    parser.add_argument("--event-start-interval", type=int, default=event_start_interval)

    event_end_interval = 5  # n second
    parser.add_argument("--event-end-interval", type=int, default=event_end_interval)

    event_duration = 10
    parser.add_argument("--event-duration", type=int, default=event_duration)

    fps = 30
    parser.add_argument("--fps", type=int, default=fps)

    width = 1280
    parser.add_argument("--width", type=int, default=width)

    height = 720
    parser.add_argument("--height", type=int, default=height)

    save = True
    parser.add_argument("--save", action="store_true", default=save)

    show = False
    parser.add_argument("--show", action="store_true", default=show)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
