"""
Just experimenting with keypoints matching
For now results are not very promising
"""

import argparse
import numpy as np
import cv2

MIN_MATCH_COUNT = 10


def match_by_features(image, template):
    """
    Function that applies SIFT detector on input
    images to extract keypoints. Then, FlannBasedMatcher tries
    to match features from both images and find similarities

    :param image (np.array): Input image
    :param template (np.array): Input template
    :return: img3 (np.array), resulting image with keypoints matching visualized
    """
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(template, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=500)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15.0)
        matchesMask = mask.ravel().tolist()
        h, w = image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv2.perspectiveTransform(pts, M)
        image = cv2.polylines(image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2,
    )
    img3 = cv2.drawMatches(image, kp1, template, kp2, good, None, **draw_params)
    return img3


def main(args):
    template = cv2.imread(args["template"])
    image = cv2.imread(args["image"])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    vis_image = match_by_features(image, template)

    cv2.imshow("Result", vis_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--template", required=True, help="Path to template image"
    )
    parser.add_argument(
        "-i",
        "--image",
        required=True,
        help="Path to image where template will be matched",
    )
    args = vars(parser.parse_args())
    main(args)
