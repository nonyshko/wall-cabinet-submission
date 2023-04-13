import numpy as np
import argparse
import imutils
import cv2


VAL_THRESHOLD = 5e7


def find_best_matching_scale(image, template):
    """
    Function that tries different scales of original
    image to find best match candidates with provided template
    
    :param image (np.array): Input image
    :param template (np.array): Input template
    :return: bbox (tuple), best_scaled_image (np.array)
    """
    
    (tH, tW) = template.shape[:2]
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    best_scaled_image = None
    # loop over the scales of the image
    # here we upscale image from x2 to x10 in 20 steps.
    for scale in np.linspace(2, 10, 20):
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        print(resized.shape)
        r = gray.shape[1] / float(resized.shape[1])
        
        # apply template matching to find the template in the image
        print(f"scale: {scale}")
        print(f"shape template: {template.shape}")
        print(f"shape resized: {resized.shape}")
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        print(f"Matchtemplate maxVal: {maxVal}")
        print("===============")
        if maxVal < VAL_THRESHOLD:
            continue

        # check to see if the iteration should be visualized
        if args.get("visualize", False):
            # draw a bounding box around the detected region
            clone = np.dstack([resized, resized, resized])
            cv2.rectangle(
                clone,
                (maxLoc[0], maxLoc[1]),
                (maxLoc[0] + tW, maxLoc[1] + tH),
                (0, 0, 255),
                2,
            )
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)
        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
            best_scaled_image = resized
    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    bbox = ((startX, startY), (endX, endY))

    return bbox, best_scaled_image


def main(args):
    template = cv2.imread(args["template"])
    image = cv2.imread(args["image"])

    # bbox returned in case additional final visualization is needed
    bbox, best_scaled_image = find_best_matching_scale(image, template)

    out_path = "image_scaled.jpg"
    cv2.imwrite(out_path, best_scaled_image)
    print(f"Successfully created scaled image for next step. Filename: {out_path}")


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
    parser.add_argument(
        "-v",
        "--visualize",
        default=0,
        help="Optional flag if debug visualization is needed",
    )
    
    args = vars(parser.parse_args())
    main(args)
