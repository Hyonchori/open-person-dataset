# Check bounding box annotation
import os
import json
from pathlib import Path

import cv2


SPLITS = {1: "Training", 2: "Validation"}
LABELS = {1: "BBOX", 2: "SEG"}
CASES = {
    "BBOX": {1: "Average_stature", 2: "Mobility_stature", 3: "Robotic_pets", 4: "Short_stature"},
    "SEG": {1: "Average_stature", 2: "Robotic_pets"}
}
SITUATIONS = {
    1: "in_balcony", 2: "in_bathroom", 3: "in_bedroom", 4: "in_elevator", 5: "in_kitchen", 6: "in_living_room",
    7: "in_meeting_room", 8: "in_office", 9: "in_stairs", 10: "in_storage", 11: "in_utility_room", 12: "in_yard",
    13: "out_Bridge", 14: "out_Building_area", 15: "out_Market", 16: "out_Park", 17: "out_Residential_area",
    18: "out_School"
}


def get_dirs_in_root(tmp_root):
    return [x for x in os.listdir(tmp_root) if os.path.isdir(os.path.join(tmp_root, x))]


def view_annot(root, target_split=None, target_label=None, target_case=None, target_sit=None, target_sit_idx=None,
               target_vid_idx=None, view_size=(1280, 720)):
    splits = get_dirs_in_root(root)
    if target_split is not None:
        assert target_split in SPLITS, f"Given target split '{target_split}' is not in {SPLITS}."
        splits = [SPLITS[target_split]]
    for split in splits:
        split_dir_path = os.path.join(root, split)
        labels = get_dirs_in_root(split_dir_path)
        if target_label is not None:
            assert target_label in LABELS, f"Given target label '{target_label}' is not in {LABELS}."
            labels = [LABELS[target_label]]
        for label in labels:
            label_dir_path = os.path.join(split_dir_path, label)
            cases = get_dirs_in_root(label_dir_path)
            if target_case is not None:
                assert target_case in CASES[label], f"Given target case '{target_case}' is not in {CASES[label]}."
                cases = [x for x in cases if CASES[label][target_case] in x]
            for case in cases:
                case_dir_path = os.path.join(label_dir_path, case)
                sits = get_dirs_in_root(case_dir_path)
                if target_sit is not None:
                    assert target_sit in SITUATIONS, f"Given target situation '{target_sit}' is not in {SITUATIONS}."
                    sits = [x for x in sorted(sits) if SITUATIONS[target_sit] in x and x.startswith("[원천]")]
                    if target_sit_idx is not None:
                        assert len(sits) >= 0 and 0 <= target_sit_idx < len(sits), \
                            f"Given video idx '{target_sit_idx}' is not valid."
                        sits = [sits[target_sit_idx]]
                for sit in sits:
                    tmp_in_dir = "_".join(sit.split("_")[1:])
                    sit_dir_path = os.path.join(case_dir_path, sit, tmp_in_dir)
                    vids = sorted(get_dirs_in_root(sit_dir_path))
                    if target_vid_idx is not None:
                        assert len(vids) >= 0 and 0 <= target_vid_idx < len(vids), \
                            f"Given video idx '{target_vid_idx}' is not valid."
                        vids = [vids[target_vid_idx]]
                    for vid in vids:
                        vid_dir_path = os.path.join(sit_dir_path, vid)
                        annot_dir_path = vid_dir_path.replace("[원천]", "[라벨]")
                        annot_path = os.path.join(annot_dir_path, vid.replace(".mp4", ".json"))
                        view_one_vid(vid_dir_path, annot_path, resize=view_size)


def view_one_vid(vid_dir_path, annot_path, txt_org=(10, 50), font_size=3, font_thickness=3, resize=(1280, 720)):
    print(f"\n--- Processing {vid_dir_path}")
    imgs = sorted(os.listdir(vid_dir_path))
    with open(annot_path) as f:
        annots = json.load(f)["annotation"]["annotations"]
        annots = sorted(annots, key=lambda annot: annot["atchOrgFileName"])
    for img_name, annot in zip(imgs, annots):
        print("\n---")
        img_path = os.path.join(vid_dir_path, img_name)
        img = cv2.imread(img_path)
        print(img_name)
        bboxes = annot["box"]
        for bbox in bboxes:
            if bbox["category_name"] != "person":
                continue
            print(bbox)
            xyxy = bbox2xyxy(bbox["box"])
            img = cv2.rectangle(img, xyxy[:2], xyxy[2:], [0, 0, 255], 2)

        img = cv2.resize(img, dsize=(resize))
        cv2.imshow("img", img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(0) & 0xFF == ord('s'):
            print("save!")


def bbox2xyxy(bbox):
    xyxy = [int(bbox["x"]),
            int(bbox["y"]),
            int(bbox["x"] + bbox["w"]),
            int(bbox["y"] + bbox["h"])]
    return xyxy


if __name__ == "__main__":
    root = "/media/daton/Data/datasets/1인칭 시점 보행영상"
    target_split = 1
    target_label = 1
    target_case = 1
    target_sit = 14
    target_sit_idx = None
    target_vid_idx = None
    view_size = (1280, 720)
    view_annot(root,
               target_split=target_split,
               target_label=target_label,
               target_case=target_case,
               target_sit=target_sit,
               target_sit_idx=target_sit_idx,
               target_vid_idx=target_vid_idx,
               view_size=view_size)
