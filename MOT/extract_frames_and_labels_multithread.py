# Extract frames by interval from images
# MOT17 => total frame: 108, total obj: 2092 with interval 50 visibility threshold 0.1
# MOT20 => total frame: 62, total obj: 8760 with interval 150 visibility threshold 0.0
import argparse
import os
import re
import glob
from pathlib import Path

import cv2
import numpy as np


SELECTION = {1: "MOT17", 2: "MOT20"}
CLASSES = {1: "pedestrian", 2: "person on vehicle", 3: "car", 4: "bicycle", 5: "motorbike", 6: "non motorized vehicle",
           7: "static person", 8: "distractor", 9: "occluder", 10: "occluder on the ground", 11: "occluder full",
           12: "reflection", 13: "crowd"}


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path



def main(args):
    root = args.root
    select = args.select
    save_dir = args.save_dir
    run_name = args.run_name
    save_labels = args.save_labels
    target_idx = args.target_idx
    target_class = args.target_class
    target_size = args.target_size
    interval = args.interval
    visibility_thr = args.visibility_thr
    view = args.view
    view_label = args.view_label
    view_visibility = args.view_visibility
    view_info = args.view_info
    save = args.save
    skip_ignored = args.skip_ignored
    clip_bbox = args.clip_bbox

    save_dir = increment_path(Path(save_dir) / run_name, exist_ok=False)
    img_save_dir = str(save_dir / "images")
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)
        os.mkdir(img_save_dir)
        if save_labels:
            gt_save_dir = str(save_dir / "labels")
            os.mkdir(gt_save_dir)

    target_mot = SELECTION[select]
    vid_root = os.path.join(root, target_mot, "train")
    if select == 1:  # MOT17
        vid_list = list(set(["-".join(x.split("-")[:-1]) for x in os.listdir(vid_root)]))
    else:  # MOT20
        vid_list = os.listdir(vid_root)
    if target_idx is not None:
        target_vids = [f"{target_mot}-{idx:02d}" for idx in target_idx]
        assert all(x in vid_list for x in target_vids), f"Some videos in {target_vids} not in {vid_list}!"
        vid_list = target_vids if select == 2 else [f"{target_vid}-DPM" for target_vid in target_vids]
    else:
        vid_list = vid_list if select == 2 else [f"{x}-DPM" for x in vid_list]

    total_frame_cnt = 0
    total_obj_cnt = 0
    for vid_name in vid_list:
        print(f"\n--- Extract frames from {vid_name}")
        vid_path = os.path.join(vid_root, vid_name, "img1")
        annot_path = os.path.join(vid_root, vid_name, "gt", "gt.txt")
        with open(annot_path) as f:
            annot = [x.replace("\n", "").split(",") for x in f.readlines()]
            track_annot = {}
            for track_info in annot:
                frame_number = int(track_info[0])
                bbox = list(map(int, track_info[2: 6]))
                conf = int(track_info[6])
                cls = int(track_info[7])
                visibility = float(track_info[8])
                tmp_info = bbox + [conf, cls, visibility]
                if frame_number in track_annot:
                    track_annot[frame_number].append(tmp_info)
                else:
                    track_annot[frame_number] = [tmp_info]

            # annotation format: [frame num, id, tl_l, tl_t, w, h, conf, class, visibility]
            # class: {1: pedestrian, 2: person on vehicle, 3: car. 4: bicycle, 5: motorbike, 6: non motorized vehicle,
            #         7: static person, 8: distractor, 9: occluder, 10: occluder on the ground, 11: occluder full,
            #         12: reflection, 13: crowd}

        imgs = sorted(os.listdir(vid_path))
        frame_cnt = 0
        obj_cnt = 0
        for i, img_name in enumerate(imgs):
            if i % interval != 0:
                continue
            else:
                frame_cnt += 1
            img_num = int(img_name.split(".")[0])
            tmp_annot = track_annot[img_num]
            img_path = os.path.join(vid_path, img_name)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            is_resized = False
            if target_size is not None:
                if target_size != [w, h]:
                    img, ratio, (dw, dh) = letterbox(img, target_size[::-1], auto=False)
                    is_resized = True
            txt = ""
            imv = img.copy()
            for track_info in tmp_annot:
                xywh = track_info[:4]
                if is_resized:
                    xywh = resize_xywh(xywh, ratio, dw, dh)
                conf = track_info[4]
                cls = track_info[5]
                visibility = track_info[6]
                if (visibility < visibility_thr) or (skip_ignored and conf == 0) or (target_class is not None and cls not in target_class):
                    continue
                else:
                    obj_cnt += 1
                if clip_bbox:
                    if target_size is not None:
                        xyxy = xywh2xyxyc(xywh, target_size[0], target_size[1])
                    else:
                        xyxy = xywh2xyxyc(xywh, w, h)
                else:
                    xyxy = xywh2xyxy(xywh)
                cpwhn = xyxy2cpwhn(xyxy, w, h)
                txt += f"{cls} {cpwhn[0]} {cpwhn[1]} {cpwhn[2]} {cpwhn[3]}\n"
                color = colors(cls, True)
                cv2.rectangle(imv, (int(cpwhn[0] * w - cpwhn[2] * w / 2), int(cpwhn[1] * h - cpwhn[3] * h / 2)),
                              (int(cpwhn[0] * w + cpwhn[2] * w / 2), int(cpwhn[1] * h + cpwhn[3] * h / 2)), color, 2)
                if view_label:
                    label = f"{CLASSES[cls]}" if not view_visibility else f"{CLASSES[cls]}: {visibility:.2f}"
                    txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
                    txt_bk_color = [int(c * 0.7) for c in color]
                    cv2.rectangle(imv, xyxy[:2], (xyxy[0] + txt_size[0] + 1, xyxy[1] + int(txt_size[1] * 1.5)),
                                  txt_bk_color, -1)
                    cv2.putText(imv, label, (xyxy[0], xyxy[1] + int(txt_size[1] * 1.2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            if view_info:
                info = f"{vid_name}: {i + 1} / {len(imgs)}"
                font_scale = int(3 * img.shape[1] / 1280)
                plot_label(imv, info, font_size=font_scale, font_thickness=font_scale)
            if view:
                cv2.imshow(vid_name, imv)
                cv2.waitKey(0)
            if save:
                img_save_path = os.path.join(img_save_dir, f"{vid_name}_{img_name}")
                cv2.imwrite(img_save_path, img)
                if save_labels:
                    gt_save_path = os.path.join(gt_save_dir, img_name.split(".")[0] + ".txt")
                    with open(gt_save_path, "w") as f:
                        f.write(txt)
        cv2.destroyAllWindows()
        print(f"\tframe: {frame_cnt}, obj: {obj_cnt}")
        total_frame_cnt += frame_cnt
        total_obj_cnt += obj_cnt
    print(f"\ntotal frame: {total_frame_cnt}, total obj: {total_obj_cnt} with interval {interval}" +
          f" visibility threshold {visibility_thr}")


def resize_xywh(xywh, ratio, dw, dh):
    xywh = [int(xywh[0] * ratio[0] + dw),
            int(xywh[1] * ratio[1] + dh),
            int(xywh[2] * ratio[0]),
            int(xywh[3] * ratio[1])]
    return xywh


def xywh2xyxyc(xywh, w, h):
    xyxy = [min(max(0, xywh[0]), w),
            min(max(0, xywh[1]), h),
            min(max(0, xywh[0] + xywh[2]), w),
            min(max(0, xywh[1] + xywh[3]), h)]
    return xyxy


def xywh2xyxy(xywh):
    xyxy = [xywh[0],
            xywh[1],
            xywh[0] + xywh[2],
            xywh[1] + xywh[3]]
    return xyxy


def xyxy2cpwhn(xyxy, img_w, img_h):
    cpwhn = [round((xyxy[0] + xyxy[2]) / 2 / img_w, 6),
             round((xyxy[1] + xyxy[3]) / 2 / img_h, 6),
             round((xyxy[2] - xyxy[0]) / img_w, 6),
             round((xyxy[3] - xyxy[1]) / img_h, 6)]
    return cpwhn


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


def plot_label(img, label, font_size=1, font_thickness=1):
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] * 2), [0, 0, 0], -1)
    cv2.putText(img, label, (5, int(label_size[1] * 1.5))
                , cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/mot"
    parser.add_argument("--root", type=str, default=root)
    parser.add_argument("--select", type=int, default=1)   # 1: MOT17, 2: MOT20

    save_dir = "/media/daton/Data/datasets/mot/extracted_frames"
    parser.add_argument("--save-dir", type=str, default=save_dir)

    run_name = "exp"
    parser.add_argument("--run-name", type=str, default=run_name)

    save_labels = True
    parser.add_argument("--save-labels", type=str, default=save_labels)

    interval = 50
    parser.add_argument("--interval", type=int, default=interval)

    target_idx = [5]
    target_idx = None
    parser.add_argument("--target-idx", type=str, default=target_idx)  # video idx list: [1, 2, 3, ...]

    target_cls = [1, 2, 7]
    parser.add_argument("--target-class", type=int, default=target_cls)  # object class list: [1, 2, 3, ...]

    target_size = [1280, 720]
    #target_size = None
    parser.add_argument("--target-size", type=int, default=target_size)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--visibility-thr", type=float, default=0.1)
    parser.add_argument("--view", action="store_true", default=False)
    parser.add_argument("--view-label", action="store_true", default=False)
    parser.add_argument("--view-visibility", action="store_true", default=True)
    parser.add_argument("--view-info", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--skip-ignored", action="store_true", default=False)
    parser.add_argument("--clip-bbox", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
