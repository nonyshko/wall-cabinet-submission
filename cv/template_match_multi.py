import argparse
import cv2
import numpy as np
import pandas as pd
from imutils import object_detection

THRESHOLD = 0.7
OUT_CSV_FNAME = "matched_objects_coords.csv"
OUT_IMAGENAME = "multi_template_matches.jpg"


def match_multiple_templates(image, template):
    """
    Function takes input image and template and tries to find multiple matches.
    Also, NMS (Non-maximum Suppression) applied to filter out overlapping
    match candidates
    :param image (np.array): Input image
    :param template (np.array): Input template
    :return: bboxes (np.array)
    """
    w, h = template.shape[::-1]

    # Apply template matching
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    # Threshold the result
    (y_points, x_points) = np.where(res >= THRESHOLD)

    bboxes = []
    for x, y in zip(x_points, y_points):
        bboxes.append([x, y, x + w, y + h])
    bboxes = np.array(bboxes)
    print(f"bboxes len before NMS: {len(bboxes)}")
    # Use NMS technique to reduce number of overlapping detected boxes
    bboxes = object_detection.non_max_suppression(bboxes)
    print(f"bboxes len after NMS: {len(bboxes)}")
    return bboxes


def main(args):
    img = cv2.imread(args["image"])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(args["template"])
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    bboxes = match_multiple_templates(img_gray, template_gray)
    # Visualize and save image for review
    for bbox in bboxes:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)

    df = pd.DataFrame(bboxes)
    df.columns = ["x0", "y0", "x1", "y1"]
    df.to_csv(OUT_CSV_FNAME)

    cv2.imwrite(OUT_IMAGENAME, img)
    print(f"Number of matched objects: {len(bboxes)}")
    print(
        f"Done processing. CSV filename: {OUT_CSV_FNAME}; Debug image filename: {OUT_IMAGENAME}"
    )


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
