# Extract frames by push "s" button from input video
import argparse
import os
import time
from pathlib import Path

import cv2


def plot_label(img, label, font_size=1, font_thickness=1):
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] * 2), [0, 0, 0], -1)
    cv2.putText(img, label, (5, int(label_size[1] * 1.5))
                , cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)


def main(args):
    target_vids = args.target_videos
    save_dir = args.save_dir
    target_size = args.target_size
    save = args.save

    if os.path.isfile(target_vids):
        vid_names = [target_vids]
    elif os.path.isdir(target_vids):
        vid_names = [x for x in os.listdir(target_vids) if os.path.join(target_vids, x) and x.endswith(".mp4")]
    assert len(vid_names) >= 1, f"Can't find target videos!"

    if not os.path.isdir(save_dir) and save:
        os.makedirs(save_dir)

    for vid_name in vid_names:
        vid_path = os.path.join(target_vids, vid_name)
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n--- Processing {vid_name}", time.sleep(0.1))
        cnt = 0
        while True:
            tmp_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, img = cap.read()
            if not ret:
                break
            imv = img.copy()
            label = f"{Path(vid_name).name}: {tmp_frame} / {total_frames}"
            plot_label(imv, label)
            cv2.imshow("img", imv)
            keyboard_input = cv2.waitKey(0) & 0xFF
            if keyboard_input == ord('s'):
                save_path = os.path.join(save_dir, Path(vid_name).name.replace(".mp4", "") + f"_{tmp_frame}.png")
                if save:
                    img = cv2.resize(img, dsize=target_size)
                    cv2.imwrite(save_path, img)
                print(f"\t{tmp_frame + 1} frame image is saved!")
                cnt += 1
            elif keyboard_input == ord('q'):
                break
            elif keyboard_input == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif keyboard_input in [81, 97]:
                # left direction
                target_frame = max(0, tmp_frame - 10)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            elif keyboard_input in [83, 100]:
                # right direction
                target_frame = min(total_frames, tmp_frame + 10)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            else:
                continue
        time.sleep(0.1)
        print(f"\ttotal {cnt} frames are saved from {Path(vid_name).name}!")


def parse_args():
    parser = argparse.ArgumentParser()

    target_vids = "/media/daton/Data/datasets/PTAW/PTAW_Datasets/PTAW172Real/extracted_videos"
    parser.add_argument("--target-videos", type=str, default=target_vids)

    save_dir = "/media/daton/Data/datasets/PTAW/PTAW_Datasets/PTAW172Real/extracted_frames"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    target_size = [1280, 720]
    parser.add_argument("--target-size", type=int, default=target_size)

    parser.add_argument("--save", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
