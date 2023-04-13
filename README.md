# Wall cabinest detection submission

As discussed, straightforward approach of using OpenCV's template matching will not work because of diffrent sizes, scales and aspect ratios of objects.

During my research I've tried following approaches:
1. Template matching using scaling approach. Idea is that object in template has higher dimensions than on big floor plan, so try to match sizes of objects.
    * First tried to downscale a **template** but faced with vanising information as picture get smaller and smaller - and thus bad matching results
    * After that I've tried to upscale an **image** and compare matching results for each scale factor. In that way I've saved upscaled floor plan for next step
2. Match multiple template instances on image. For that I've took upscaled image from previous step and performed matching with template and decided to leave results with matching score above some defined *THRESHOLD*. After that used NMS (Non-maximum Suppression) to filter out overlapping match candidates.
3. Features matching using SIFT and ORB algorithms. These algorithms are used to calculate keypoints and descriptors (features) of images. After that matching of respectful features from both images was performed. This experiment showed poor performance at first stage so I left it as is. Possible tuning and improving of this part in future.
4. Deep Learning with One Shot Object Detector (OSOD). Idea that I don't have such kind of dataset and wanted to experiment with just one image. Tried with Python library developed by Apple: *turicreate* - a collection of simple pretrained machine learning models. But training on single image with number of iterations = 500 did not show any results. Tried training on bigger number of iterations = 5000 went to crash inside library. Didn't want to spend more time as this approach is experimental and may take a lot of time to have any kind of results.

## Usage
Tested on Ubuntu 22.04 with Python 3.10.6
1. Install requirement libraries 

    `$pip install -r requirements.txt`
2. Generate upscaled floor plan image

    `$cd cv`

    `$python template_match_scale.py -i ../data/floor_plan.jpg -t ../data/objects/wall_cabinets/0.png`
3. Calc number of template objects on image and generate .csv file with coordinates of objects as well as image with visualized objects.

    `$python template_match_multi.py -i image_scaled.jpg -t ../data/objects/wall_cabinets/0.png`

### NOTES
* Each of wall cabinets processed separately as there is a small change in some params to work correctly. E.g. for obj0 THRESHOLD is 0.7 and for obj1 THRESHOLD is 0.45 

* This approach has obvious limitations because of mentioned above concerns. Also, check results for object6: *data/samples/multi_template_matches_obj6.jpg* and object itself: *data/objects/wall_cabinets/6.png*. For that specific case scripts failed to produce some acceptable results.