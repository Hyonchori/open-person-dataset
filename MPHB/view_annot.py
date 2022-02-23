# Multiple Pose Human Body Database (LSP/MPII-MPHB)
# http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/MPHB.html
# writen on 2022.01.25

import os

import cv2
from openpyxl import load_workbook


def read_action_xlsx(action_label_root, target_action):
    action_list = [x.split(".")[0] for x in os.listdir(action_label_root) if x.endswith(".xlsx")]
    assert target_action in action_list, f"Given '{target_action}' is not in {action_list}"
    action_file_path = os.path.join(action_label_root, target_action + ".xlsx")
    wb = load_workbook(action_file_path)["Sheet1"]
    target_imgs = []
    for row in wb.rows:
        for cell in row:
            target_imgs.append(f"{cell.value}.jpg")
    return target_imgs


def get_target_imgs(img_dir, action_label_root, target_action=None):
    if target_action is None:
        target_imgs = os.listdir(img_dir)
    else:
        target_imgs = read_action_xlsx(action_label_root, target_action)
    return target_imgs


def read_bbox_label(bbox_label_root):
    label_path = os.path.join(bbox_label_root, "MPHB-label.txt")
    with open(label_path) as f:
        raw_label = f.readlines()
    bbox_labels = {}
    tmp_bboxes = []
    for raw in raw_label:
        if raw.startswith("idx"):
            idx = int(raw.split(":")[-1][:-1])
        elif raw.startswith("bbox"):
            continue
        elif raw.startswith("\n"):
            bbox_labels[idx] = tmp_bboxes
            tmp_bboxes = []
        elif raw.startswith("source"):
            continue
        else:
            bbox = [int(float(x)) for x in raw[:-1].split()]
            tmp_bboxes.append(bbox)
    return bbox_labels


def plot_bboxes(bboxes, img):
    for bbox in bboxes:
        cv2.rectangle(img, bbox[:2], bbox[2:], [0, 0, 255])


def view_annot(img_dir, bbox_label_root, action_label_root, target_action=None):
    target_imgs = get_target_imgs(img_dir, action_label_root, target_action)
    bbox_labels = read_bbox_label(bbox_label_root)
    for target_img_name in target_imgs:
        target_img_path = os.path.join(img_dir, target_img_name)
        if not os.path.isfile(target_img_path):
            continue
        target_img_num = int(target_img_name.split(".")[0])
        target_bboxes = bbox_labels[target_img_num]
        print(f"\n--- {target_img_name}")
        print(target_bboxes)
        target_img = cv2.imread(target_img_path)
        plot_bboxes(target_bboxes, target_img)
        img_name = target_action if target_action is not None else "image"
        cv2.imshow(img_name, target_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    root = "/media/daton/Data/datasets/MPHB"
    img_dir = os.path.join(root, "Human Body Image")
    bbox_label_root = os.path.join(root, "MPHB-label-txt")
    action_label_root = os.path.join(root, "label-linux-compress")

    target_action = "lying"
    #target_action = None
    view_annot(img_dir, bbox_label_root, action_label_root, target_action)
