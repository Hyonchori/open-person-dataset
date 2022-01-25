import os

import cv2
from tqdm import tqdm
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


def xyxy2cpwhn(xyxy, w, h):
    cpwhn = [(xyxy[0] + xyxy[2]) / 2 / w,
             (xyxy[1] + xyxy[3]) / 2 / h,
             (xyxy[2] - xyxy[0]) / w,
             (xyxy[3] - xyxy[1]) / h]
    return cpwhn


def export_images_and_labels(img_dir, bbox_label_root, action_label_root, output_dir, target_action=None):
    out_img_dir = os.path.join(output_dir, "images")
    out_label_dir = os.path.join(output_dir, "labels")
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)
    if not os.path.isdir(out_label_dir):
        os.makedirs(out_label_dir)
    target_imgs = get_target_imgs(img_dir, action_label_root, target_action)
    bbox_labels = read_bbox_label(bbox_label_root)
    target_action = target_action if target_action is not None else "All"
    print(f"\n--- Exporting {target_action} images to YOLOv5 format")
    for target_img_name in tqdm(target_imgs):
        target_img_path = os.path.join(img_dir, target_img_name)
        if not os.path.isfile(target_img_path):
            continue
        target_img_num = int(target_img_name.split(".")[0])
        xyxys = bbox_labels[target_img_num]
        target_img = cv2.imread(target_img_path)
        h, w, _ = target_img.shape
        target_label = ""
        for xyxy in xyxys:
            cpwhn = xyxy2cpwhn(xyxy, w, h)
            target_label += f"0 {cpwhn[0]} {cpwhn[1]} {cpwhn[2]} {cpwhn[3]}\n"
        cv2.imwrite(os.path.join(out_img_dir, target_img_name), target_img)
        with open(os.path.join(out_label_dir, target_img_name.split(".")[0] + ".txt"), "w") as f:
            f.write(target_label)


if __name__ == "__main__":
    root = "/media/daton/Data/datasets/MPHB"
    img_dir = os.path.join(root, "Human Body Image")
    bbox_label_root = os.path.join(root, "MPHB-label-txt")
    action_label_root = os.path.join(root, "label-linux-compress")

    target_action = "lying"
    output_dir = os.path.join(root, "custom", target_action)
    export_images_and_labels(img_dir, bbox_label_root, action_label_root, output_dir, target_action)