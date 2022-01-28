import os
import json

import cv2


def view_2d_annot(image_root, annot_2d_root,
                  target_action=None, target_num=None, target_case=None, target_cam=None):
    action_dict = {int(x.split("_")[-1]): x for x in os.listdir(image_root)
                   if os.path.isdir(os.path.join(image_root, x))}
    if target_action is not None:
        assert target_action in action_dict, f"Given target action '{target_action}' not in {sorted(action_dict.keys())}"
        action_dict = {target_action: action_dict[target_action]}

    for action in action_dict:
        action_dir_path = os.path.join(image_root, action_dict[action])
        num_dict = {int(x.split("-")[-1]): x for x in os.listdir(action_dir_path)
                    if os.path.isdir(os.path.join(action_dir_path, x))}
        if target_num is not None:
            assert target_num in num_dict, f"Given target number '{target_num}' not in {sorted(num_dict.keys())}"
            num_dict = {target_num: num_dict[target_num]}

        for num in num_dict:
            num_dir_path = os.path.join(action_dir_path, num_dict[num])
            while len(os.listdir(num_dir_path)) == 1:
                num_dir_path = os.path.join(num_dir_path, os.listdir(num_dir_path)[0])
            case_dict = {}
            for case in os.listdir(num_dir_path):
                if not os.path.isdir(os.path.join(num_dir_path, case)):
                    continue
                case_idx = int(case.split("_")[-1].split("-")[0])
                if case_idx in case_dict:
                    case_dict[case_idx].append(case)
                else:
                    case_dict[case_idx] = [case]
            if target_case is not None:
                assert target_case in case_dict, f"Given target case '{target_case}' not in {sorted(case_dict.keys())}"
                case_dict = {target_case: case_dict[target_case]}

            for case_idx in case_dict:
                cam_dict = {int(x.split("-")[-1][1:]): x for x in case_dict[case_idx]}
                if target_cam is not None:
                    if target_cam not in cam_dict:
                        continue
                    else:
                        cam_dict = {target_cam: cam_dict[target_cam]}

                for cam in cam_dict:
                    cam_dir_path = os.path.join(num_dir_path, cam_dict[cam])
                    target_annot_dir = os.path.join(annot_2d_root, cam_dict[cam].split("_")[0])
                    target_annot_path = os.path.join(target_annot_dir, f"{cam_dict[cam]}_2D.json")
                    with open(target_annot_path) as f:
                        target_annot = json.load(f)
                    visualize_one_vid(cam_dir_path, target_annot)


def visualize_one_vid(img_dir_path, annot, txt_org=(30, 50), font_size=3, font_thickness=3):
    label = img_dir_path.split("/")[-1]
    print(f"\n--- Showing {label}")
    imgs = sorted(os.listdir(img_dir_path))
    bboxes = annot["annotations"]
    for img_name, bbox in zip(imgs, bboxes):
        img_path = os.path.join(img_dir_path, img_name)
        img = cv2.imread(img_path)
        bbox = bbox["bbox"]
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0, 0, 255], 2)
        txt_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)
        cv2.rectangle(img, (txt_org[0], txt_org[1] + 5),
                      (txt_org[0] + txt_size[0], txt_org[1] - txt_size[1] - 5), [0, 0, 0], -1)
        cv2.putText(img, label, txt_org, cv2.FONT_HERSHEY_PLAIN, font_size, [255, 255, 255], font_thickness,
                    cv2.LINE_AA)
        cv2.imshow("img", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    root = "/media/daton/Data/datasets/사람동작 영상"  # in Ubuntu
    annot_root = os.path.join(root, "annotation")
    annot_2d_root = os.path.join(annot_root, "Annotation_2D_tar", "2D")
    image_root = os.path.join(root, "이미지")

    target_action = 6
    target_num = 3
    target_case = None
    target_cam = 1
    # actions = {3: "앉기", 6: "쓰러짐", 14: "윗몸 일으키기",}
    view_2d_annot(image_root, annot_2d_root,
                  target_action=target_action,
                  target_num=target_num,
                  target_case=target_case,
                  target_cam=target_cam)
