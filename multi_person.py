import cv2
import numpy as np
import os
import sys
from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("person", type=int, default=1, help="Number of person you want to extract/ default is 1")
parser.add_argument("-v", "--video", help="If video file is the input", action="store_true")
parser.add_argument("path", type=str, help="path of the input image or video")
parser.add_argument("background_path", type=str, help="path of the desired background image")
args = parser.parse_args()

# Load the pre-trained model data
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Change the config infermation
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1
    
config = InferenceConfig()

# COCO dataset object names
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# This function is used to place object into the new background image.
# image[:,:,c_channel] means: image[:,:,0] is the Blue channel,image[:,:,1] is the Green channel, image[:,:,2] is the Red channel
# mask_arr == 1 means that these pixels belongs to the object.
# np.where function work: In the background image, if the pixel belong to object, change it to object-pixel-values to place object into the new background.
def apply_new_background(image, mask):
    # desired background image
    background_image = str(args.background_path) #'demo/background_empty_road.jpg'
    # reading the background image
    background_img = cv2.imread(background_image)
    # changing the size of background image bacuase it should be equal to original image
    size = (image.shape[1], image.shape[0])
    resized_background_img = cv2.resize(background_img, size, interpolation = cv2.INTER_AREA)
    # Showing the desired background for visulization
    cv2.imshow('Desired_Background', resized_background_img)
    
    for i in range(len(mask)):
        mask_list = mask[i]
        mask_arr = np.array(mask_list)
        for c_channel in range(3):
            resized_background_img[:, :, c_channel] = np.where(
                mask_arr == 1,
                image[:, :, c_channel],
                resized_background_img[:, :, c_channel]
            )

    return resized_background_img


# This function is just a part of the next function (change_background)
def area_element(area_n_index):
    return area_n_index[0]

# This function is used to show the object detection result in original image.
def change_background(image, boxes, masks, ids, names, scores, desired_n_person = 1):
    # max_area will save the largest object (a person) for all the detection results
    area = []
    # Initialising the mask variable that will carry out the largest object (person) mask
    #largest_mask, second_largest_mask = masks[:,:]
    mask = []
    # n_instances saves the amount of all objects
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    n_person = 0
    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue
        # use label to select person object from all the 80 classes in COCO dataset
        label = names[ids[i]]
        if label == 'person':
            # compute the square of each person-object
            y1, x1, y2, x2 = boxes[i]
            square = (y2 - y1) * (x2 - x1)
            area.insert(n_person, (square, i))
            n_person+=1
        else:
            continue
    
    if not area:
        desired_n_person = 0
    elif len(area)<desired_n_person:
        desired_n_person = len(area)
    
    sorted_area = sorted(area, reverse=True, key=area_element)
    for i in range(desired_n_person):
        mask.insert(i, list(masks[:,:,sorted_area[i][1]]))

    # apply new background for the image
    image = apply_new_background(image, mask)
        
    return image

# If input is the video-file
if args.video:
    # Input video path
    input_video = str(args.path) #"demo/abcd2_dharmesh_trim.mp4"
    # passing the video file into openCV Video capture object
    capture = cv2.VideoCapture(input_video)
    # these 2 lines can be removed if you dont have a 1080p camera.
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Recording Video
    fps = capture.get(5)
    width = int(capture.get(3))
    height = int(capture.get(4))
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter("demo_results_multi/saved_video.avi", fcc, fps, (width, height))

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = change_background(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], desired_n_person = args.person
        )
        # Showing each processed frame
        cv2.imshow('video', frame)

        # Recording Video
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

# If input is the image-file
else:
    # Input the original image name
    original_image = str(args.path) #'demo/test_data/tree/File3.jpg'
    image = cv2.imread(original_image)
    # Keeping the copy of original image as test image for visulization
    test_image = cv2.imread(original_image)

    results = model.detect([image], verbose=0)
    r = results[0]
    frame = change_background(
        image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
    )

    # Showing the Original and final images for visulization
    cv2.imshow('Original_Image', test_image)
    cv2.imshow('Output_Image', frame)

    # Wait for keys to exit (by pressing the esc key) or save
    key = cv2.waitKey(0)
    if key == 27:                 
        cv2.destroyAllWindows()
    elif key == ord('s'):        
        cv2.imwrite('saved_image.jpg', frame)
        cv2.destroyAllWindows()

