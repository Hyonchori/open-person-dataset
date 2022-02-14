import os

import xmltodict


action_list = {1: "Loitering", 2: "Intrusion", 3: "Abandonment", 4: "FireDetection", 5: "Violence", 6: "Falldown"}
target_action = action_list[2]


if __name__ == "__main__":
    root = "/media/daton/SAMSUNG/4. 민간분야(2021 특수환경)"
    data_root = os.path.join(root, "distribution")

    vids = [x for x in sorted(os.listdir(data_root))
            if x.endswith(".mp4") and os.path.isfile(os.path.join(data_root, x))]
    annots = [x.replace(".mp4", ".xml") for x in vids
              if os.path.isfile(os.path.join(data_root, x.replace(".mp4", ".xml")))]

    target_vid_list = []
    for i, annot_name in enumerate(annots):
        annot_path = os.path.join(data_root, annot_name)
        with open(annot_path) as f:
            annot = xmltodict.parse(f.read())["KisaLibraryIndex"]["Library"]
        scenario = annot["Scenario"]
        if scenario == target_action:
            target_vid_list.append(vids[i])
            start_time = annot["Clip"]["Alarms"]["Alarm"]["StartTime"]
            print(f"\n--- {vids[i]}")
            print(start_time)
