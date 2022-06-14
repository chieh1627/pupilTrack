import cv2
import numpy as np



def pupilTrack(src: np.ndarray, gamma: float, minArea: int, maxArea: int) -> np.ndarray:
    """
        This program is used to detect the pupil part and output the detection result.
        :param src: np.ndarray -> input image.
        :param gamma: float -> value of Gamma Correction.
        :param minArea: int -> Prediction of the minimum area of the pupil.
        :param maxArea: int -> Prediction of the maximum area of the pupil.
        :return: output: np.ndarray - > output image with mask.
    """
    img = src.copy()

    # Step1. Color to Gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step2. use GaussianBlur to reduce noise.
    img = cv2.GaussianBlur(src=img, ksize=(5, 5), sigmaX=2)

    # Step3. Gamma Correction
    img_gamma = np.power(img.copy() / 255.0, gamma)
    img_gamma = img_gamma * 255.0
    img_gamma = img_gamma.astype(np.uint8)

    # Step4. Binarization
    ret, _ = cv2.threshold(img_gamma, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img_threshold = cv2.threshold(img_gamma, ret * 0.8, 255, cv2.THRESH_BINARY)

    # Step5. Find the contour and check if it is the pupil part
    contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    area = ([cv2.contourArea(contour) for contour in contours])

    idx = -1
    for i in range(len(area)):
        if minArea <= area[i] <= maxArea:
            ellipse = cv2.fitEllipse(contours[i])
            axis_l, axis_s = max(ellipse[1]), min(ellipse[1])
            if area[i] != np.max(area) and 0.4 <= axis_s / axis_l <= 1:
                idx = i

    # Step6. Add a mask to the pupil.
    output = np.zeros(shape=src.shape, dtype=np.uint8)
    mask = np.zeros(shape=img.shape, dtype=np.uint8)

    if idx == -1:
        return output, 0.0
    else:
        ellipse = cv2.fitEllipse(contours[idx])
        mask = cv2.ellipse(mask, ellipse, 255, -1, cv2.LINE_AA)

        output[:, :, 0] = mask
        output[:, :, 2] = mask
        return output, 1.0
