import cv2
import os
import numpy as np
from tqdm import tqdm

from pupil import pupilTrack

path = "./final/dataset/public"

d = 'S5'
s_ = [f"{i + 1:02d}" for i in range(26)]

for s in tqdm(s_):
    os.makedirs(f"./S5_solution/{s}", exist_ok=True)
    files_len = len(os.listdir(f"{path}/{d}/{s}"))
    x = [cv2.imread(f"{path}/{d}/{s}/{idx}.jpg") for idx in range(files_len)]

    mask = np.zeros(shape=(480, 640, 3)).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if s == "01" or s == "02":
        out = cv2.VideoWriter(f'output_{s}.mp4', fourcc, 30.0, (720, 170))

    elif s == "03":
        out = cv2.VideoWriter(f'output_{s}.mp4', fourcc, 30.0, (600, 220))

    elif s == "04" or s == "05" or s == "06" or s == "07" or s == "08" or s == "09":
        out = cv2.VideoWriter(f'output_{s}.mp4', fourcc, 30.0, (600, 250))

    elif s == "10" or s == "11" or s == "12" or s == "13":
        out = cv2.VideoWriter(f'output_{s}.mp4', fourcc, 30.0, (980, 360))

    elif s == "14" or s == "15" or s == "16" or s == "17" or s == "18" or s == "19" or s == "20" or s == "21" or s == "22":
        out = cv2.VideoWriter(f'output_{s}.mp4', fourcc, 30.0, (700, 300))

    elif s == "23" or s == "24" or s == "25" or s == "26":
        out = cv2.VideoWriter(f'output_{s}.mp4', fourcc, 30.0, (1180, 380))

    with open(f"./conf.txt", "w") as f:
        for i in range(files_len):
            if s == "01" or s == "02":
                img = x[i][100:270, 240:600, :]
                output, conf = pupilTrack(src=img, gamma=0.1, minArea=500, maxArea=6000)
                mask[100:270, 240:600, :] = output
                out.write(np.hstack([img, output]))

            elif s == "03":
                img = x[i][130:350, 260:560, :]
                output, conf = pupilTrack(src=img, gamma=0.1, minArea=500, maxArea=6000)
                mask[130:350, 260:560, :] = output
                out.write(np.hstack([img, output]))

            elif s == "04" or s == "05" or s == "06" or s == "07" or s == "08" or s == "09":
                img = x[i][130:380, 260:560, :]
                output, conf = pupilTrack(src=img, gamma=0.1, minArea=500, maxArea=6000)
                mask[130:380, 260:560, :] = output
                out.write(np.hstack([img, output]))

            elif s == "10" or s == "11" or s == "12" or s == "13":
                img = x[i][120:480, 150:640, :]
                output, conf = pupilTrack(src=img, gamma=0.1, minArea=500, maxArea=6000)
                mask[120:480, 150:640, :] = output
                out.write(np.hstack([img, output]))

            elif s == "14" or s == "15" or s == "16" or s == "17" or s == "18" or s == "19" or s == "20" or s == "21" or s == "22":
                img = x[i][100:400, 150:500, :]
                output, conf = pupilTrack(src=img, gamma=0.1, minArea=500, maxArea=6000)
                mask[100:400, 150:500, :] = output
                out.write(np.hstack([img, output]))

            elif s == "23" or s == "24" or s == "25" or s == "26":
                img = x[i][100:480, 50:640, :]
                output, conf = pupilTrack(src=img, gamma=0.1, minArea=500, maxArea=6000)
                mask[100:480, 50:640, :] = output
                out.write(np.hstack([img, output]))

            cv2.imwrite(f'./S5_solution/{s}/{i}.png', mask)
            f.write(f"{conf}\n")

    out.release()