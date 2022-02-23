import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class RoI:
    def __init__(self, img_size, window_name):
        self.img_size = img_size
        self.window_name = window_name
        self.ref_img = np.zeros(img_size, np.uint8)
        self.roi_pts = []
        self.roi_check = False
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.on_mouse)

    def on_mouse(self, event, x, y, *kwargs):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.roi_check:
                self.roi_check = False
                self.roi_pts = []
                self.ref_img = np.zeros(self.img_size, np.uint8)
            elif not self.roi_check and len(self.roi_pts) == 0:
                pass
            else:
                self.roi_check = True
                self.roi_pts = np.array(self.roi_pts, np.int32)

    def imshow(self, input_img):
        roi_pts = self.roi_pts
        if not self.roi_check:
            for pt in roi_pts:
                cv2.circle(input_img, (pt[0], pt[1]), 2, (0, 225, 225), 2)
        else:
            cv2.fillPoly(self.ref_img, [roi_pts.reshape((-1, 1, 2))], (0, 225, 225))
        result = cv2.addWeighted(input_img, 1, self.ref_img, 0.5, 0)
        cv2.imshow(self.window_name, result)


def main(args):
    vid_path = args.vid_path
    save = args.save
    event_type = args.event_type
    just_view = args.just_view

    assert os.path.isfile(vid_path), f"Given video path '{vid_path}' is wrong!"
    vid_cap = cv2.VideoCapture(vid_path)
    _, tmp_img = vid_cap.read()
    img_size = tmp_img.shape
    window_name = Path(vid_path).name
    roi = RoI(img_size=img_size, window_name=window_name)
    vid_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n--- Processing {vid_path}")
    for _ in tqdm(range(vid_frames)):
        ret, img = vid_cap.read()
        if not ret:
            break
        roi.imshow(img)
        if just_view:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
        else:
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    if save and not just_view:
        save_name = window_name.replace(".mp4", ".json")
        save_path = vid_path.replace(window_name, save_name)
        if len(roi.roi_pts) != 0:
            roi_pts = roi.roi_pts.tolist() if isinstance(roi.roi_pts, np.ndarray) else roi.roi_pts
        else:
            roi_pts = [(0, 0), (img_size[1], 0), (img_size[1], img_size[0]), (0, img_size[0])]
        roi_pts_norm = [(round(pt[0] / img_size[1], 6), round(pt[1] / img_size[0], 6)) for pt in roi_pts]
        roi_info = {
            "vid_name": window_name,
            "width": img_size[1],
            "height": img_size[0],
            "name": event_type,
            "detection_area": roi_pts,
            "detection_area_norm": roi_pts_norm
        }
        with open(save_path, "w") as f:
            json.dump(roi_info, f)
            print(f"\nroi information is saved in {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    vid_path = "/home/daton/Desktop/gs/loitering_gs/KISA_loitering_13.mp4"
    vid_path = "/home/daton/Desktop/gs/intrusion_gs/KISA_intrusion_20.mp4"
    vid_path = "/media/daton/SAMSUNG/4. 민간분야(2021 특수환경)/distribution/C123200_009.mp4"
    parser.add_argument("--vid-path", type=str, default=vid_path)

    save = True
    parser.add_argument("--save", action="store_true", default=save)

    event_type = "loitering"
    parser.add_argument("--event-type", type=str, default=event_type)

    just_view = False
    parser.add_argument("--just-view", action="store_true", default=just_view)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
