# input_data: directory path including images, output_data: selected images by keyboard input 's'
import os
import cv2


def select_frames(img_dir, save_dir=None):
    assert os.path.isdir(img_dir), f"Given {img_dir} is wrong path"
    if save_dir is None:
        save_dir = img_dir + "_selected"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    imgs = os.listdir(img_dir)
    for i, img_name in enumerate(imgs):
        print(f"\n--- {i + 1}/{len(imgs)} {img_name}")
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        if cv2.waitKey(0) & 0xFF == ord('s'):
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, img)
            print("save!")


if __name__ == "__main__":
    #root = "/home/daton/pedestrian-detection-in-hazy-weather/dataset/hazy_person/PICTURES_LABELS_TRAIN/PICTURES"
    root = "/home/daton/pedestrian-detection-in-hazy-weather/dataset/hazy_person/PICTURES_LABELS_TEMP_TEST/PICTURES"
    output_dir = None
    select_frames(root, output_dir)
