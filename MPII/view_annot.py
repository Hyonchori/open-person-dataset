import os

from scipy import io
import pandas as pd


if __name__ == "__main__":
    root = "/media/daton/Data/datasets/MPII"
    label_dir = os.path.join(root, "mpii_human_pose_v1_u12_2")

    mat_path = os.path.join(label_dir, "mpii_dataset.csv")
    #mat = io.loadmat(mat_path)
    df = pd.read_csv(mat_path)
    print(df)
    print(df.columns)
    print(df["Activity"].unique())
    print(df["Category"].unique())
