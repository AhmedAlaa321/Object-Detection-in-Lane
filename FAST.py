import cv2
import numpy as np
import urllib.request

def resize(img):
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent/ 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def region_of_interest(frame):
    roi = frame[300:1500,0:1000]
    return roi

def object_detection(frame):
    frame = resize(frame)
    frame = region_of_interest(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([161, 155, 84])
    upper_red = np.array([179, 255, 255])
    mask = cv2.inRange(hsv,lower_red,upper_red)
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(mask, None)
    print("Number of keypoints Detected: ", len(keypoints))
    image = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return image

def stream_show():
    url='http://172.28.129.48:8080/shot.jpg'
    while True:
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
        img = cv2.imdecode(imgNp,-1)
        image = object_detection(img)
        cv2.imshow('ObjectDetection - FAST', image)
        if ord('q') == cv2.waitKey(10):
            exit(0)

stream_show()