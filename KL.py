# %%
import os
import cv2
import numpy as np
from pupil import pupilTrack
from tqdm import tqdm

path = "./bonus"
d_ = ["KL"]

for d in d_:
    os.makedirs(f"./{d}", exist_ok=True)
    files_len = len(os.listdir(f"{path}/{d}"))
    x = [cv2.imread(f"{path}/{d}/{idx:04d}.jpg").astype(np.uint8) for idx in range(files_len)]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'output_{d}.mp4', fourcc, 30.0, (640, 320))

for i in tqdm(range(len(x))):
    mask = np.zeros(shape=(320, 320, 3)).astype(np.uint8)

    img = x[i][100:200, 100:200, :].astype(np.uint8)
    output, conf = pupilTrack(src=img, gamma=0.2, minArea=100, maxArea=1000)
    mask[100:200, 100:200, :] = output
    w = np.hstack([x[i], mask]).astype(np.uint8)
    cv2.imwrite(f'./{d}/{d}_{i + 1:02d}.png', w)
    out.write(w)

out.release()