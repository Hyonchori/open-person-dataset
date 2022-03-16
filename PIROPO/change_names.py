import os
import cv2
from tqdm import tqdm

root = "/media/daton/Data/datasets/PIROPO/extracted_frames/exp2"
save_dir = "/media/daton/Data/datasets/PIROPO/extracted_frames/exp"

imgs = os.listdir(root)
for img_name in tqdm(imgs):
    img_path = os.path.join(root, img_name)
    img = cv2.imread(img_path)

    save_img_name = " ".join(img_name.split(".")[:-1]) + ".png"
    save_path = os.path.join(save_dir, save_img_name)
    cv2.imwrite(save_path, img)
