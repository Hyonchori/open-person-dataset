# View bounding box annotations from COCO dataset
import argparse
import os

import cv2
import numpy as np
from pycocotools.coco import COCO

from coco_80_classes import COCO_CLASSES


SELECTION = {1: "train2017", 2: "val2017"}


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


def main(args):
    root = args.root
    select = args.select
    target_cls = args.target_class
    view_size = args.view_size
    ignore_crowd = args.ignore_crowd
    hide_info = args.hide_info
    hide_label = args.hide_labels

    target_selection = SELECTION[select]
    annot_dir = os.path.join(root, "annotations_trainval2017", "annotations")
    annot_path = os.path.join(annot_dir, f"instances_{target_selection}.json")
    if os.path.isfile(annot_path):
        coco = COCO(annot_path)
    else:
        raise Exception(f"annotations file is not found")

    if target_cls is None:
        target_img_ids = coco.getImgIds()
    else:
        cat_ids = coco.getCatIds(catNms=[COCO_CLASSES[c] for c in target_cls])
        target_img_ids = coco.getImgIds(catIds=cat_ids)
    target_imgs = coco.loadImgs(target_img_ids)
    img_dir = os.path.join(root, target_selection)
    for target_img in target_imgs:
        img_idx = target_img["id"]
        img_file = target_img["file_name"]
        width = target_img["width"]
        height = target_img["height"]
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)

        is_resized = False
        if view_size is not None:
            if view_size != [width, height]:
                img, ratio, (dw, dh) = letterbox(img, view_size[::-1], auto=False)
                is_resized = True

        ann_id = coco.getAnnIds(imgIds=img_idx)
        anns = coco.loadAnns(ann_id)
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id - 1 not in target_cls:
                continue
            iscrowd = ann["iscrowd"]
            if ignore_crowd and iscrowd == 1:
                continue
            bbox = ann["bbox"]
            if is_resized:
                bbox = resize_xywh(bbox, ratio, dw, dh)
            xyxy = xywh2xyxy(bbox)
            color = colors(cat_id, True)
            cv2.rectangle(img, xyxy[:2], xyxy[2:], color, 2)
        cv2.imshow(img_file, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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

    root = "/media/daton/Data/datasets/coco"
    parser.add_argument("--root", type=str, default=root)
    parser.add_argument("--select", type=int, default=2)  # 1: train2017, 2: val2017

    target_cls = [0]
    #target_cls = None
    parser.add_argument("--target-class", type=int, default=target_cls)

    view_size = [1280, 720]
    parser.add_argument("--view-size", type=int, default=view_size)

    parser.add_argument("--ignore-crowd", action="store_true", default=False)
    parser.add_argument("--hide_info", action="store_true", default=False)
    parser.add_argument("--hide-labels", action="store_true", default=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
