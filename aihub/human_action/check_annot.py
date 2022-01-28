import os
import json

import cv2


actions = {
    3: "앉기",
    6: "쓰러짐",
    14: "윗몸 일으키기",
}


def view_annot(img_dir, annot):
    print(img_dir)
    imgs = os.listdir(img_dir)
    print(len(imgs))
    print(annot.keys())
    bboxes = annot["annotations"]
    print(len(bboxes))
    for img_name, bbox in zip(imgs, bboxes):
        img_path = os.path.join(img_dir, img_name)
        print(img_path)
        img_path = r"E:/datasets/사람동작 영상/이미지/image_action_14/image_14-1/14-1/14-1_001-C08/14-1_001-C08_009.jpg"
        print(img_path)
        print(os.path.isfile(img_path))
        img = cv2.imread(img_path)
        print(img.shape)

        action_cls = int(img_name.split("_")[0].split("-")[0])
        cv2.imshow(actions[action_cls], img)
        cv2.waitKey(0)
        break



if __name__ == "__main__":
    root = "E:\datasets\사람동작 영상"
    annot_root = os.path.join(root, "annotation")
    annot_root_2d = os.path.join(annot_root, "Annotation_2D_tar", "2D")
    annot_2d_list = [x for x in os.listdir(annot_root_2d) if os.path.isdir(os.path.join(annot_root_2d, x))]

    image_root = os.path.join(root, "이미지")
    image_list = [x for x in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, x))]

    # target action check
    target_action = 14
    action_dir_path = os.path.join(image_root, f"image_action_{target_action}")

    # target number
    target_number= 1
    num_dict = {int(x.split("-")[-1]): x for x in os.listdir(action_dir_path)
                if os.path.isdir(os.path.join(action_dir_path, x))}
    num_dir_path = os.path.join(action_dir_path, num_dict[target_number])
    if len(os.listdir(num_dir_path)) == 1:
        num_dir_path = os.path.join(num_dir_path, f"{target_action}-{target_number}")

    # target case
    target_case = 1
    case_dict = {}
    for case in os.listdir(num_dir_path):
        if not os.path.isdir(os.path.join(num_dir_path, case)):
            continue
        case_idx = int(case.split("-")[-2].split("_")[-1])
        if case_idx in case_dict:
            case_dict[case_idx].append(case)
        else:
            case_dict[case_idx] = [case]

    # target camera
    target_cam_idx = min(6, len(case_dict[target_case]))
    target_cam = case_dict[target_case][target_cam_idx]
    target_img_dir = os.path.join(num_dir_path, target_cam)

    # target annotation
    target_annot_dir = os.path.join(annot_root_2d, target_cam.split("_")[0])
    target_annot_2d_path = os.path.join(target_annot_dir, f"{target_cam}_2D.json")
    with open(target_annot_2d_path) as f:
        target_annot = json.load(f)

    view_annot(target_img_dir, target_annot)