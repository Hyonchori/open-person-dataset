# View 2D annotations of '사람동작 영상' dataset
import argparse
import os
import json
from pathlib import Path

import cv2

ACTIONS = {6: "image_action_6"}


def main(args):
    root = args.root
    target_action = args.target_action
    target_action_num = args.target_action_num
    target_scene = args.target_scene
    target_cam = args.target_cam
    view_size = args.view_size

    annot2d_dir_path = os.path.join(root, "annotation", "Annotation_2D_tar", "2D")
    annots = [x for x in os.listdir(annot2d_dir_path) if os.path.isdir(os.path.join(annot2d_dir_path, x))]

    img_dir_path = os.path.join(root, "이미지")
    actions = [x for x in os.listdir(img_dir_path)
               if os.path.isdir(os.path.join(img_dir_path, x)) and x in ACTIONS.values()]
    if target_action is not None:
        assert all(x in ACTIONS for x in target_action), f"Some elements of '{target_action}' are not in {ACTIONS}"
        assert all(ACTIONS[x] in actions for x in target_action), \
            f"Some elements of '{[ACTIONS[x] in actions for x in target_action]}' are not in {actions}"
        actions = [ACTIONS[x] for x in target_action]

    for action in actions:
        action_dir_path = os.path.join(img_dir_path, action)
        action_nums_dict = {int(x.split("-")[-1]): x for x in os.listdir(action_dir_path)
                            if os.path.isdir(os.path.join(action_dir_path, x))}
        if target_action_num is not None:
            assert all(x in action_nums_dict for x in target_action_num), \
                f"Some elements of '{target_action_num}' not in {action_nums_dict}"
            action_nums_dict = {x: action_nums_dict[x] for x in target_action_num}

        for action_num in action_nums_dict.values():
            action_num_dir_path = os.path.join(action_dir_path, action_num)
            while len(os.listdir(action_num_dir_path)) == 1:
                action_num_dir_path = os.path.join(action_num_dir_path, os.listdir(action_num_dir_path)[0])
            scenes_dict = {}
            for scene in os.listdir(action_num_dir_path):
                scene_num = int(scene.split("_")[-1].split("-")[0])
                if scene_num in scenes_dict:
                    scenes_dict[scene_num].append(scene)
                else:
                    scenes_dict[scene_num] = [scene]
            if target_scene is not None:
                assert all(x in scenes_dict for x in target_scene), \
                    f"Some elements of '{target_scene}' not in {scenes_dict}"
                scenes_dict = {x: scenes_dict[x] for x in target_scene}
            annot_dir_path = os.path.join(annot2d_dir_path, action_num.split("_")[-1])

            for scene in scenes_dict.values():
                cam_dict = {int(x.split("-")[-1][1:]): x for x in scene}
                if target_cam is not None:
                    assert all(x in cam_dict for x in target_cam), \
                        f"Some elements of '{target_cam}' not in {cam_dict}"
                    cam_dict = {x: cam_dict[x] for x in target_cam}

                for cam in cam_dict.values():
                    cam_dir_path = os.path.join(action_num_dir_path, cam)
                    annot_path = os.path.join(annot_dir_path, cam + "_2D.json")
                    visualize_one_vid(cam_dir_path, annot_path, view_size)


def visualize_one_vid(img_dir_path, annot_path, view_size, font_size=2, font_thickness=2):
    print(f"\n--- Processing {img_dir_path}")
    cam_name = Path(img_dir_path).name
    action_idx = int(cam_name.split("-")[0])
    if action_idx == 6:
        label = "falldown"
    else:
        label = ""
    imgs = sorted(os.listdir(img_dir_path))
    with open(annot_path) as f:
        annot = json.load(f)
    target_indices = [i for i, x in enumerate(annot["images"]) if Path(x["img_path"]).name in imgs]
    for i, (img_name, target_idx) in enumerate(zip(imgs, target_indices)):
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)

        xyxy = list(map(int, annot["annotations"][target_idx]["bbox"]))
        color = [0, 255, 255]
        img = cv2.rectangle(img, xyxy[:2], xyxy[2:], color, 2)
        plot_label(img, xyxy, label, color)

        if view_size is not None:
            img = cv2.resize(img, dsize=view_size)
        info = f"{cam_name}: {i + 1}/{len(imgs)}"
        plot_info(img, info)
        cv2.imshow("img", img)
        cv2.waitKey(1)


def plot_info(img, info, font_size=2, font_thickness=2):
    label_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] * 2), [0, 0, 0], -1)
    cv2.putText(img, info, (5, int(label_size[1] * 1.5))
                , cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)


def plot_label(img, xyxy, label, color, font_size=1, font_thickness=2):
    txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
    txt_bk_color = [int(c * 0.7) for c in color]
    cv2.rectangle(img, xyxy[:2], (xyxy[0] + txt_size[0] + 1, xyxy[1] + int(txt_size[1] * 1.5)),
                  txt_bk_color, -1)
    cv2.putText(img, label, (xyxy[0], xyxy[1] + int(txt_size[1] * 1.2)),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)


def parse_args():
    parser = argparse.ArgumentParser()

    root = "/media/daton/Data/datasets/사람동작 영상"
    parser.add_argument("--root", type=str, default=root)

    target_action = [6]
    # target_action = None
    parser.add_argument("--target-action", type=str, default=target_action)

    target_action_num = [3]
    # target_action_num = None
    parser.add_argument("--target-action-num", type=str, default=target_action_num)

    target_scene = [3]
    # target_scene = None
    parser.add_argument("--target-scene", type=str, default=target_scene)

    target_cam = [3]
    target_cam = None
    parser.add_argument("--target-cam", type=str, default=target_cam)

    view_size = [1280, 720]
    parser.add_argument("--view-size", type=int, default=view_size)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
