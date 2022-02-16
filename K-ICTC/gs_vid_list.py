# state: [in/out, out_night/out_day, out_rainy/out_sunny, num]

loitering_list = {
    "C001101_003": [1, 1, 1, 1],
    "C001201_004": [1, 1, 1, 2],
    "C002201_004": [1, 1, 1, 1],
    "C004301_003": [0, 0, 0, 1],
    "C005101_002": [1, 1, 1, 1],
    "C006101_002": [1, 0, 1, 1],
    "C007101_001": [1, 1, 1, 1],
    "C007201_002": [1, 1, 1, 3],
    "C007201_005": [1, 1, 1, 1],
    "C008201_002": [1, 1, 1, 1],
    "C008301_005": [1, 1, 1, 1],
    "C017101_001": [0, 0, 0, 1],
    "C045100_001": [1, 1, 1, 1],
    "C045100_002": [1, 1, 1, 2],
    "C045100_003": [1, 0, 1, 1],
    "C045100_004": [1, 0, 1, 2],
    "C045300_003": [1, 0, 1, 1],
    "C045300_004": [1, 0, 1, 2],
    "C047100_006": [1, 1, 1, 3],
    "C047100_008": [1, 1, 1, 2],
    "C049100_002": [1, 1, 1, 2],
    "C049100_005": [1, 0, 1, 3],
    "C049100_011": [1, 1, 1, 3],
    "C049200_005": [1, 0, 1, 3],
    "C050100_005": [1, 1, 1, 1],
    "C051100_001": [1, 1, 1, 1],
    "C051100_003": [1, 1, 1, 2],
    "C051100_004": [1, 0, 1, 1],
    "C051100_005": [1, 0, 1, 2],
    "C052100_002": [1, 1, 1, 2],
    "C052100_005": [1, 0, 1, 2],
    "C052100_017": [1, 1, 1, 2],
    "C052300_013": [1, 1, 1, 1],
    "C054100_002": [1, 1, 1, 2],
    "C054100_005": [1, 0, 1, 2],
    "C055200_001": [1, 0, 1, 1],
    "C055200_005": [1, 1, 1, 2],
    "C055200_011": [1, 1, 0, 1],
    "C056200_002": [1, 1, 1, 2],
    "C056200_004": [1, 0, 1, 1],
    "C058100_002": [1, 1, 1, 2],
    "C058100_004": [1, 0, 1, 1],
    "C058200_008": [1, 1, 1, 2],
    "C082100_003": [1, 1, 0, 1],
    "C086100_005": [1, 1, 0, 1],
    "C087100_006": [1, 1, 0, 1],
    "C099100_005": [1, 1, 0, 2],
    "C104300_032": [1, 1, 0, 1],
    "aihub_subway_cctv_1": [0, 0, 0, 4],
    "aihub_subway_cctv_2": [0, 0, 0, 7],
    "aihub_subway_cctv_3": [0, 0, 0, 18],
    "aihub_subway_cctv_4": [0, 0, 0, 9]
}
print(len(loitering_list))
out_count = 0
out_night = 0
out_day = 0
out_rain = 0
out_sunny = 0
_in = 0
for v in loitering_list.values():
    if v[0]:
        out_count += v[3]
        if v[1]:
            out_day += v[3]
        else:
            out_night += v[3]
        if v[2]:
            out_sunny += v[3]
        else:
            out_rain += v[3]
    else:
        _in += v[3]
print(f"\n--- 배회 데이터셋 객체 구성: 총 {out_count + _in}")
print(f"야외: {out_count}, 실내: {_in}")
print(f"야외 야간: {out_night}, 야외 주간: {out_day}")
print(f"야외 우천시: {out_rain}, 야외 맑음: {out_sunny}")
print("\n-----------------------------------------\n")

# state: [in/out, out_night/out_day, out_rainy/out_sunny, num]

intrusion_list = {
    "C001202_001": [1, 1, 1, 1],
    "C001302_004": [1, 1, 1, 2],
    "C002202_001": [1, 1, 1, 1],
    "C002202_003": [1, 1, 1, 1],
    "C003102_001": [1, 0, 1, 1],
    "C013102_002": [1, 1, 1, 2],
    "C013202_003": [1, 1, 1, 1],
    "C016202_011": [1, 1, 1, 1],
    "C016302_007": [1, 1, 1, 1],
    "C019102_001": [1, 1, 1, 1],
    "C021102_001": [1, 1, 1, 1],
    "C045101_002": [1, 1, 1, 1],
    "C045101_003": [1, 0, 1, 1],
    "C045101_004": [1, 0, 1, 1],
    "C045101_006": [1, 1, 1, 1],
    "C045101_009": [1, 1, 1, 1],
    "C045101_010": [1, 0, 1, 1],
    "C045201_011": [1, 1, 1, 1],
    "C045301_004": [1, 0, 1, 1],
    "C045301_006": [1, 1, 1, 1],
    "C050101_011": [1, 0, 1, 2],
    "C050101_013": [1, 1, 1, 1],
    "C050101_014": [1, 1, 1, 2],
    "C050101_015": [1, 1, 1, 3],
    "C050201_010": [1, 0, 1, 1],
    "C050301_009": [1, 1, 1, 3],
    "C050301_010": [1, 0, 1, 1],
    "C050301_011": [1, 0, 1, 2],
    "C055101_005": [1, 1, 1, 2],
    "C055101_007": [1, 1, 1, 1],
    "C055201_002": [1, 0, 1, 2],
    "C055201_005": [1, 1, 1, 2],
    "C055201_007": [1, 1, 1, 1],
    "C055301_001": [1, 0, 1, 1],
    "C055301_007": [1, 1, 1, 1],
    "C055301_010": [1, 1, 1, 1],
    "C058101_002": [1, 1, 1, 2],
    "C058101_013": [1, 1, 1, 1],
    "C058301_001": [1, 1, 1, 1],
    "C058301_010": [1, 1, 1, 1],
    "C058301_015": [1, 1, 1, 2],
    "C059100_001": [0, 0, 0, 10],
    "C061102_003": [0, 0, 0, 14],
    "C065100_009": [0, 0, 0, 10],
    "C082101_002": [1, 1, 0, 1],
    "C082201_005": [1, 1, 0, 1],
    "C090201_003": [1, 0, 0, 1],
    "C092101_001": [1, 0, 0, 1],
    "C092201_002": [1, 1, 0, 1],
    "C098201_004": [1, 1, 0, 3],
    "C104301_001": [1, 0, 0, 3],
    "C106201_017": [1, 1, 0, 1],
    "C106301_024": [1, 1, 0, 1],
    "C114101_001": [1, 0, 0, 2]
}
print(len(intrusion_list))
out_count = 0
out_night = 0
out_day = 0
out_rain = 0
out_sunny = 0
_in = 0
for v in intrusion_list.values():
    if v[0]:
        out_count += v[3]
        if v[1]:
            out_day += v[3]
        else:
            out_night += v[3]
        if v[2]:
            out_sunny += v[3]
        else:
            out_rain += v[3]
    else:
        _in += v[3]
print(f"\n--- 침입 데이터셋 구성: {out_count + _in}")
print(f"야외: {out_count}, 실내: {_in}")
print(f"야외 야간: {out_night}, 야외 주간: {out_day}")
print(f"야외 우천시: {out_rain}, 야외 맑음: {out_sunny}")
