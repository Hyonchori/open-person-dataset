import os
import argparse
from datetime import timedelta

import xmltodict


DEV_OR_CIVIL = {3: "3. 연구개발분야", 4: "4. 민간분야(2021 특수환경)"}
CLASSES = {1: "배회", 2: "침입", 3: "유기", 4: "쓰러짐", 5: "싸움", 6: "방화", 7: "줄서기", 8: "카운팅"}
EVENTS = {1: "Loitering", 2: "Intrusion", 3: "Abandonment", 4: "Falldown", 5: "Violence", 6: "FireDetection",
          7: "Queueing", 8: "PeopleCounting"}


def main(args):
    kisa_root = args.kisa_root
    dev_or_civil = args.dev_or_civil
    target_class = args.target_class

    if dev_or_civil not in DEV_OR_CIVIL:
        print(f"Given verbose '{dev_or_civil}' not in '[3, 4]'. Default verbose(4) is set. {DEV_OR_CIVIL[4]}")
        dev_or_civil = 4
    target_event = [EVENTS[x] for x in target_class if x in CLASSES]
    print(f"\nTarget event: {target_event}")
    assert len(target_event) >= 1, "Target event is None. Revise arguments."

    target_dir = os.path.join(kisa_root, DEV_OR_CIVIL[dev_or_civil])
    if dev_or_civil == 4:
        vid_dir = os.path.join(target_dir, "distribution")
        target_vids = [x for x in os.listdir(vid_dir) if os.path.isfile(os.path.join(vid_dir, x)) and
                       (x.endswith(".mp4"))]
    else:  # elif dev_or_civil == 3:
        target_dir = os.path.join(target_dir, "1. 해외환경(1500개)")
        target_vids = []
        for tmp_class in target_class:
            tmp_class = CLASSES[tmp_class]
            vid_dir = os.path.join(target_dir, [x for x in os.listdir(target_dir) if tmp_class in x][0])
            target_vids += [x for x in os.listdir(vid_dir) if os.path.isfile(os.path.join(vid_dir, x)) and
                            (x.endswith(".mp4"))]
    assert len(target_vids) >= 1, "Target videos is None. Revise arguments."
    target_annots = [x.replace(".mp4", ".xml") for x in target_vids]

    valid_vids = []
    print(f"\n--- Searching target videos in {vid_dir}")
    for target_vid, target_annot in zip(target_vids, target_annots):
        annot_path = os.path.join(vid_dir, target_annot)
        gt = gt_from_annot(annot_path)
        inter = list(set(target_event) & set(gt["events"]))
        if len(inter) == 0:
            continue
        else:
            vid_path = os.path.join(vid_dir, target_vid)
            print(vid_path)
            valid_vids.append(vid_path)
    print(valid_vids)
    print(len(valid_vids))



def time2hms(t: str):
    if len(t.split(":")) == 3:
        h, m, s = [float(x) for x in t.split(":")]
        return h, m, s
    else:
        m, s = [float(x) for x in t.split(":")]
        return 0, m, s


def hms2delta(target_time):
    return timedelta(hours=target_time[0], minutes=target_time[1], seconds=target_time[2])


def gt_from_annot(annot_path: str):
    gt = {"alarms": [],
          "events": []}
    with open(annot_path) as f:
        annot = xmltodict.parse(f.read())
    clip = annot["KisaLibraryIndex"]["Library"]["Clip"]
    vid_duration = time2hms(clip["Header"]["Duration"])
    vid_duration_delta = hms2delta(vid_duration)
    alarms = clip["Alarms"]["Alarm"]
    if not isinstance(alarms, list):
        alarms = [alarms]
    for i, alarm in enumerate(alarms):
        tmp_alarm = {}
        alarm_class = alarm["AlarmDescription"]
        start_time = time2hms(alarm["StartTime"])
        start_time_delta = hms2delta(start_time)
        start_frame_rate = start_time_delta / vid_duration_delta
        if i == 0:
            gt["start_frame_rate"] = start_frame_rate
        alarm_class = alarm_class if not alarm_class == "Queuing" else "Queueing"
        if alarm_class not in ["Queueing", "PeopleCounting"]:
            duration = time2hms(alarm["AlarmDuration"])
            duration_delta = hms2delta(duration)
        else:
            duration_delta = hms2delta((0, 0, 1))
            if "Ingress" in alarm:
                tmp_alarm["Ingress"] = alarm["Ingress"]  # Queueing
            elif "Egress" in alarm:
                tmp_alarm["Egress"] = alarm["Egress"]  # Queueing
            elif "InCount" in alarm:
                tmp_alarm["InCount"] = alarm["InCount"]  # PeopleCounting
            elif "OutCount" in alarm:
                tmp_alarm["OutCount"] = alarm["OutCount"]  # PeopleCounting
        end_frame_rate = (start_time_delta + duration_delta) / vid_duration_delta
        tmp_alarm["alarm_class"] = alarm_class
        tmp_alarm["start_frame_rate"] = start_frame_rate
        tmp_alarm["end_frame_rate"] = end_frame_rate
        gt["alarms"].append(tmp_alarm)
        if alarm_class not in gt["events"]:
            gt["events"].append(alarm_class)

        if i == len(alarms) - 1:
            gt["end_frame_rate"] = end_frame_rate
    return gt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--kisa-root", type=str, default="/media/daton/SAMSUNG")
    parser.add_argument("--dev_or_civil", type=int, default=4)  # 3: 해외환경, 4: 민간
    parser.add_argument("--target_class", type=int, default=[1])

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
