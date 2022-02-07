import json

file_path = "/media/daton/Data/datasets/지하철 역사 내 CCTV 이상행동 영상/subway_cctv_coco.json"
with open(file_path) as f:
    annot = json.load(f)
print(annot)