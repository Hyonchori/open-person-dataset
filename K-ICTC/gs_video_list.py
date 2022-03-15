# state: [in/out, out_night/out_day, out_rainy/out_sunny, num]

loitering_list = [
    {"C045100_001": [1, 1, 1, 1], "C045100_002": [1, 1, 1, 2], "C045100_003": [1, 0, 1, 1], "C045100_004": [1, 0, 1, 2],
     "C045300_003": [1, 0, 1, 1], "C045300_004": [1, 0, 1, 2]},

    {"C001101_003": [1, 1, 1, 1], "C001201_004": [1, 1, 1, 2]},

    {"C007101_001": [1, 1, 1, 1], "C007201_002": [1, 1, 1, 3], "C007201_005": [1, 1, 1, 1]},

    {"C047100_006": [1, 1, 1, 3], "C047100_008": [1, 1, 1, 2]},

    {"C049100_002": [1, 1, 1, 2], "C049100_005": [1, 0, 1, 3], "C049100_011": [1, 1, 1, 3],
     "C049200_005": [1, 0, 1, 3]},

    {"C051100_001": [1, 1, 1, 1], "C051100_003": [1, 1, 1, 2], "C051100_004": [1, 0, 1, 1],
     "C051100_005": [1, 0, 1, 2]},

    {"C052100_002": [1, 1, 1, 2], "C052100_005": [1, 0, 1, 2], "C052100_017": [1, 1, 1, 2],
     "C052300_013": [1, 1, 1, 1]},

    {"C054100_002": [1, 1, 1, 2], "C054100_005": [1, 0, 1, 2]},

    {"C055200_001": [1, 0, 1, 1], "C055200_005": [1, 1, 1, 2], "C055200_011": [1, 1, 0, 1]},

    {"C056200_002": [1, 1, 1, 2], "C056200_004": [1, 0, 1, 1]},

    {"C058100_002": [1, 1, 1, 2], "C058100_004": [1, 0, 1, 1], "C058200_008": [1, 1, 1, 2]},

    {"C002201_004": [1, 1, 1, 1], "C004301_003": [0, 0, 0, 1], "C005101_002": [1, 1, 1, 1], "C006101_002": [1, 0, 1, 1],
     "C008201_002": [1, 1, 1, 1], "C008301_005": [1, 1, 1, 1], "C017101_001": [0, 0, 0, 1],
     "C050100_005": [1, 1, 1, 1]},

    {"C082100_003": [1, 1, 0, 1], "C086100_005": [1, 1, 0, 1], "C087100_006": [1, 1, 0, 1], "C099100_005": [1, 1, 0, 2],
     "C104300_032": [1, 1, 0, 1]},

    {"aihub_subway_cctv_1": [0, 0, 0, 4]},
    {"aihub_subway_cctv_2": [0, 0, 0, 7]},
    {"aihub_subway_cctv_3": [0, 0, 0, 18]},
    {"aihub_subway_cctv_4": [0, 0, 0, 9]}
]
total_cnt = 0
total_in_cnt = 0
total_out_cnt = 0
total_out_night_cnt = 0
total_out_day_cnt = 0
total_out_rain_cnt = 0
total_out_sun_cnt = 0

for i, lts in enumerate(loitering_list):
    tmp_cnt = 0
    in_cnt = 0
    out_cnt = 0
    out_night_cnt = 0
    out_day_cnt = 0
    out_rain_cnt = 0
    out_sun_cnt = 0
    for lt in lts.values():
        if lt[0] == 0:
            in_cnt += lt[-1]
            tmp_cnt += lt[-1]
        else:
            if lt[1] == 0:
                out_night_cnt += lt[-1]
            else:
                out_day_cnt += lt[-1]
            if lt[2] == 0:
                out_rain_cnt += lt[-1]
            else:
                out_sun_cnt += lt[-1]
            out_cnt += lt[-1]
            tmp_cnt += lt[-1]
    print(f"\n--- KISA loitering {i + 1}")
    print(f"\t합: {tmp_cnt}, 실내: {in_cnt}, 야외: {out_cnt} (야외-주간: {out_day_cnt}, 야외-야간: {out_night_cnt}, / "
          f"야외-우천: {out_rain_cnt}, 야외-맑음: {out_sun_cnt})")
    total_cnt += tmp_cnt
    total_in_cnt += in_cnt
    total_out_cnt += out_cnt
    total_out_night_cnt += out_night_cnt
    total_out_day_cnt += out_day_cnt
    total_out_rain_cnt += out_rain_cnt
    total_out_sun_cnt += out_sun_cnt
print(f"\n총합: {total_cnt}, 실내-합: {total_in_cnt}, 야외-합: {total_out_cnt} (야외-주간-합: {total_out_day_cnt}, "
      f"야외-야간-합: {total_out_night_cnt}, / 야외-우천-합: {total_out_rain_cnt}, 야외-맑음-합: {total_out_sun_cnt})\n")


intrusion_list = [
    {"C001202_001": [1, 1, 1, 1], "C001302_004": [1, 1, 1, 1]},

    {"C002202_001": [1, 1, 1, 1], "C002202_003": [1, 1, 1, 1]},

    {"C013102_002": [1, 1, 1, 2], "C013202_003": [1, 1, 1, 1]},

    {"C016202_011": [1, 1, 1, 1], "C016302_007": [1, 1, 1, 1]},

    {"C045101_002": [1, 1, 1, 1], "C045101_003": [1, 0, 1, 1], "C045101_004": [1, 0, 1, 1], "C045101_006": [1, 1, 1, 1],
     "C045101_009": [1, 1, 1, 1], "C045101_010": [1, 0, 1, 1], "C045201_011": [1, 1, 1, 1], "C045301_004": [1, 0, 1, 1],
     "C045301_006": [1, 1, 1, 1]},

    {"C050101_011": [1, 0, 1, 2], "C050101_013": [1, 1, 1, 1], "C050101_014": [1, 1, 1, 2], "C050101_015": [1, 1, 1, 3],
     "C050201_010": [1, 0, 1, 1], "C050301_009": [1, 1, 1, 3], "C050301_010": [1, 0, 1, 1],
     "C050301_011": [1, 0, 1, 2]},

    {"C055101_005": [1, 1, 1, 2], "C055101_007": [1, 1, 1, 1], "C055201_002": [1, 0, 1, 2], "C055201_005": [1, 1, 1, 2],
     "C055201_007": [1, 1, 1, 1], "C055301_001": [1, 0, 1, 1], "C055301_007": [1, 1, 1, 1],
     "C055301_010": [1, 1, 1, 1]},

    {"C058101_002": [1, 1, 1, 2], "C058101_013": [1, 1, 1, 1]},

    {"C058301_001": [1, 1, 1, 1], "C058301_010": [1, 1, 1, 1], "C058301_015": [1, 1, 1, 2]},

    {"C082101_002": [1, 1, 0, 1], "C082201_005": [1, 1, 0, 1]},

    {"C092101_001": [1, 0, 0, 1], "C092201_002": [1, 1, 0, 1]},

    {"C106201_017": [1, 1, 0, 1], "C106301_024": [1, 1, 0, 1]},

    {"C003102_001": [1, 0, 1, 1]},
    {"C019102_001": [1, 1, 1, 1]},
    {"C021102_001": [1, 1, 1, 1]},
    {"C090201_003": [1, 0, 0, 1]},
    {"C104301_001": [1, 0, 0, 3]},
    {"C114101_001": [1, 0, 0, 2]},
    {"C110101_004": [1, 0, 0, 3]},

    {"C059100_001": [0, 0, 0, 10]},
    {"C061102_003": [0, 0, 0, 14]},
    {"C065100_009": [0, 0, 0, 10]},
]
total_cnt = 0
total_in_cnt = 0
total_out_cnt = 0
total_out_night_cnt = 0
total_out_day_cnt = 0
total_out_rain_cnt = 0
total_out_sun_cnt = 0

for i, lts in enumerate(intrusion_list):
    tmp_cnt = 0
    in_cnt = 0
    out_cnt = 0
    out_night_cnt = 0
    out_day_cnt = 0
    out_rain_cnt = 0
    out_sun_cnt = 0
    for lt in lts.values():
        if lt[0] == 0:
            in_cnt += lt[-1]
            tmp_cnt += lt[-1]
        else:
            if lt[1] == 0:
                out_night_cnt += lt[-1]
            else:
                out_day_cnt += lt[-1]
            if lt[2] == 0:
                out_rain_cnt += lt[-1]
            else:
                out_sun_cnt += lt[-1]
            out_cnt += lt[-1]
            tmp_cnt += lt[-1]
    print(f"\n--- KISA Intrusion {i + 1}")
    print(f"\t합: {tmp_cnt}, 실내: {in_cnt}, 야외: {out_cnt} (야외-주간: {out_day_cnt}, 야외-야간: {out_night_cnt}, / "
          f"야외-우천: {out_rain_cnt}, 야외-맑음: {out_sun_cnt})")
    total_cnt += tmp_cnt
    total_in_cnt += in_cnt
    total_out_cnt += out_cnt
    total_out_night_cnt += out_night_cnt
    total_out_day_cnt += out_day_cnt
    total_out_rain_cnt += out_rain_cnt
    total_out_sun_cnt += out_sun_cnt
print(f"\n총합: {total_cnt}, 실내-합: {total_in_cnt}, 야외-합: {total_out_cnt} (야외-주간-합: {total_out_day_cnt}, "
      f"야외-야간-합: {total_out_night_cnt}, / 야외-우천-합: {total_out_rain_cnt}, 야외-맑음-합: {total_out_sun_cnt})")