import cv2
from darkflow.net.build import TFNet
import sys
from PIL import Image


options = {'model': 'cfg/tiny-yolo-voc-3c.cfg',
           'load': 3750,
           'threshold': 0.15,
           'gpu': 0.7}

tfnet = TFNet(options)

C = []  # Center
R = []  # Radius
L = []  # Label
conf = []


if len(sys.argv) > 1:
    im_name = sys.argv[1]
else:
    im_name = 'data/HRI001.jpg'

image = cv2.imread(im_name, -1)
width, height = Image.open(im_name).size
for h in range(0, height, 480):
    for w in range(0, width, 640):
        im = image[h:h + 480, w:w + 640]
        output = tfnet.return_predict(im)

        RBC = 0
        WBC = 0
        Platelets = 0

        for prediction in output:
            label = prediction['label']
            confidence = prediction['confidence']

            tl = (prediction['topleft']['x'], prediction['topleft']['y'])
            br = (prediction['bottomright']['x'], prediction['bottomright']['y'])

            height, width, _ = image.shape
            center_x = int((tl[0] + br[0]) / 2)
            center_y = int((tl[1] + br[1]) / 2)
            center = (center_x + w, center_y + h)
            radius = int((br[0] - tl[0]) / 2)

            C.append(center)
            R.append(radius)
            L.append(label)
            conf.append(confidence)

record = []

for i in range(0, len(C)):
    center = C[i]
    radius = R[i]
    label = L[i]
    confidence = conf[i]

    if label == 'RBC':
        color = (255, 0, 0)
    elif label == 'WBC':
        color = (0, 255, 0)
    elif label == 'Platelets':
        color = (0, 0, 255)

    image = cv2.circle(image, center, radius, color, 5)
    font = cv2.FONT_HERSHEY_COMPLEX
    image = cv2.putText(image, str(confidence), (center[0] - 30, center[1] + 10), font, 1, color, 2)

cv2.imwrite('outh.jpg', image)
