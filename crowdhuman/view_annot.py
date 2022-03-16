# View annotations of 'crowdhuman' dataset
import argparse
import os

import cv2
import numpy as np

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

SELECTIONS = {1: "train", 2: "val"}
CLASSES = {1: "fbox", 2: "vbox", 3: "hbox"}


def main(args):
    root = args.root
    target_select = args.target_select
    target_split = args.target_split
    target_class = args.target_class
    view_size = args.view_size
    view_label = args.view_label
    crop_bbox = args.crop_bbox

    selects_dict = {}
    img_dirs = [x for x in os.listdir(root)
                if os.path.isdir(os.path.join(root, x)) and any(v in x for v in SELECTIONS.values())]
    if target_select is not None:
        assert all(x in SELECTIONS for x in target_select), f"Some elements of {target_select} are not in {SELECTIONS}"
        tmp_select = target_select
    else:
        tmp_select = SELECTIONS
    for select in tmp_select:
        select_dirs = [x for x in img_dirs if SELECTIONS[select] in x]
        selects_dict[SELECTIONS[select]] = select_dirs

    for select, splits in selects_dict.items():
        if select == "train" and target_split is not None:
            splits = [x for x in splits if int(x[-2:]) in target_split]
            assert len(splits) == len(target_split), f"Some elements of {target_split} are wrong!"
        img_paths_dict = {}
        for split in splits:
            split_dir_path = os.path.join(root, split, "Images")
            for img_name in [x for x in os.listdir(split_dir_path) if os.path.isfile(os.path.join(split_dir_path, x))]:
                img_paths_dict[img_name] = os.path.join(split_dir_path, img_name)

        assert all(x in CLASSES for x in target_class), f"Some elements of {target_class} are wrong!"
        annot_path = os.path.join(root, f"annotation_{select}.odgt")
        visualize_images(img_paths_dict, annot_path, target_class, view_size, view_label, crop_bbox)


def visualize_images(img_paths_dict, annot_path, target_class, view_size, view_label, crop_bbox):
    with open(annot_path) as f:
        data = f.readlines()
        for d in data:
            annot = eval(d)
            img_name = f"{annot['ID']}.jpg"
            if img_name not in img_paths_dict:
                continue
            img_path = img_paths_dict[img_name]
            img = cv2.imread(img_path)
            is_resized = False
            if view_size is not None and img.shape[:2] != view_size[::-1]:
                img, ratio, (dw, dh) = letterbox(img, view_size[::-1], auto=False)
                is_resized = True
            for gtbox in annot["gtboxes"]:
                extra = gtbox["extra"]
                if "ignore" in extra:
                    ignore = bool(extra["ignore"])
                else:
                    ignore = False
                bboxes = {c: gtbox[CLASSES[c]] for c in target_class}
                for cls, bbox in bboxes.items():
                    if is_resized:
                        bbox = resize_xywh(bbox, ratio, dw, dh)
                    if crop_bbox:
                        h, w, _ = img.shape
                        xyxy = xywh2xyxyc(bbox, w, h)
                    else:
                        xyxy = xywh2xyxy(bbox)
                    color = colors(cls, True) if not ignore else [0, 0, 255]
                    img = cv2.rectangle(img, xyxy[:2], xyxy[2:], color, 3)
                    if view_label:
                        plot_label(img, xyxy, CLASSES[cls], color)

            img_info = f"{img_name}"
            plot_info(img, img_info)
            cv2.imshow("img", img)
            cv2.waitKey(0)


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


def resize_xywh(xywh, ratio, dw, dh):
    xywh = [int(xywh[0] * ratio[0] + dw),
            int(xywh[1] * ratio[1] + dh),
            int(xywh[2] * ratio[0]),
            int(xywh[3] * ratio[1])]
    return xywh


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

    root = "/media/daton/Data/datasets/crowdhuman"
    parser.add_argument("--root", type=str, default=root)

    # 1: train, 2: valid
    target_select = [1]
    # target_select = None
    parser.add_argument("--target-select", type=int, default=target_select)

    # train: [1, 2, 3], val: None
    target_split = [1, 2]
    # target_split = None
    parser.add_argument("--target-split", type=int, default=target_split)

    # 1: fbox, 2: vbox, 3: hbox
    target_class = [1]
    # target_class = None
    parser.add_argument("--target-class", type=int, default=target_class)

    view_size = [1280, 720]
    parser.add_argument("--view-size", type=int, default=view_size)

    parser.add_argument("--view-label", action="store_true", default=False)
    parser.add_argument("--crop-bbox", action="store_true", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
