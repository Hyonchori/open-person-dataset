# View annotations of 'Top-View-Multi-Person tracking dataset'
# from https://github.com/ucuapps/top-view-multi-person-tracking
import argparse
import os

import cv2
import numpy as np

SPLITS = {1: "train", 2: "test"}
CLASSES = {1: "person"}


def main(args):
    root = args.root
    target_split = args.target_split
    view_size = args.view_size

    splits = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)) and x in SPLITS.values()]
    if target_split is not None:
        assert all(x in SPLITS for x in target_split), f"Some elements of {target_split} are not in {SPLITS}"
        splits = [SPLITS[x] for x in target_split]

    for split in splits:
        split_dir_path = os.path.join(root, split, split)
        cams_dict = {}
        for file_name in os.listdir(split_dir_path):
            if not file_name.endswith("jpg"):
                continue
            cam_name = "_".join(file_name.split("_")[:-1])
            if cam_name in cams_dict:
                cams_dict[cam_name].append(file_name)
            else:
                cams_dict[cam_name] = [file_name]
        for cam_name, imgs in cams_dict.items():
            visualize_one_vid(split_dir_path, imgs, cam_name, view_size)


def visualize_one_vid(img_dir_path, imgs, vid_name, view_size):
    print(f"\n--- Processing {vid_name}")
    imgs = sorted(imgs)
    for i, img_name in enumerate(imgs):
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        is_resized = False
        if view_size is not None and [w, h] != view_size:
            img, ratio, (dw, dh) = letterbox(img, view_size[::-1], auto=False)
            is_resized = True

        annot_path = img_path.replace(".jpg", ".txt")
        print(img_path)
        print(annot_path)
        with open(annot_path) as f:
            annots = f.readlines()
        for annot in annots:
            cls, cxn, cyn, wn, hn = list(map(float, annot.split()))
            color = [0, 0, 255]
            xyxy = cpwhn2xyxy(cxn, cyn, wn, hn, w, h)
            img = cv2.rectangle(img, xyxy[:2], xyxy[2:], color, 2)
            plot_label(img, xyxy, CLASSES[int(cls + 1)], color)

        info = f"{vid_name}: {i + 1} / {len(imgs)}"
        plot_info(img, info)
        cv2.imshow("img", img)
        cv2.waitKey(0)


def cpwhn2xyxy(cxn, cyn, wn, hn, w, h):
    xyxy = [int((cxn - wn / 2) * w),
            int((cyn - hn / 2) * h),
            int((cxn + wn / 2) * w),
            int((cyn + hn / 2) * h),]
    return xyxy


def plot_label(img, xyxy, label, color, font_size=1, font_thickness=1):
    txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
    txt_bk_color = [int(c * 0.7) for c in color]
    cv2.rectangle(img, xyxy[:2], (xyxy[0] + txt_size[0] + 1, xyxy[1] + int(txt_size[1] * 1.5)),
                  txt_bk_color, -1)
    cv2.putText(img, label, (xyxy[0], xyxy[1] + int(txt_size[1] * 1.2)),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)


def plot_info(img, info, font_size=2, font_thickness=2):
    label_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] * 2), [0, 0, 0], -1)
    cv2.putText(img, info, (5, int(label_size[1] * 1.5))
                , cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)


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


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/TVMP"
    parser.add_argument("--root", type=str, default=root)

    # 1: train, 2: test
    target_split = [1]
    # target_split
    parser.add_argument("--target-split", type=int, default=target_split)

    view_size = [1280, 720]
    view_size = None
    parser.add_argument("--view-size", type=int, default=view_size)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
