from turtle import rt
import cv2
from darkflow.net.build import TFNet
import sys
from PIL import Image
from tqdm import tqdm
import numpy as np


options = {'model': 'cfg/tiny-yolo-voc-3c.cfg',
           'load': 3750,
           'threshold': 0.15,
           'gpu': 0.7}

tfnet = TFNet(options)


if len(sys.argv) > 1:
    im_name = sys.argv[1]
else:
    im_name = 'data/HRI001.jpg'

def open_image(im_name, alpha):
    img = Image.open(im_name)
    width, height = int(img.size[0] * alpha), int(img.size[1] * alpha)
    img = img.resize((width, height))
    temp = 'temp.jpg'
    img.save(temp)

    image = cv2.imread(temp, -1)
    return image, width, height

def draw_circles(image, alpha, C, R, L, conf, out_name='outh.jpg'):
    image, width, height = open_image(im_name, alpha)
    for i in range(len(C)):
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

    cv2.imwrite(out_name, image)

def predict(im_name, alpha=5):
    C = []  # Center
    R = []  # Radius
    L = []  # Label
    conf = []

    image, width, height = open_image(im_name, alpha)
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
    return C, R, L, conf

def merge(C_list, R_list, L_list, conf_list):
    C, R, L, conf = np.concatenate(C_list), np.concatenate(R_list), np.concatenate(L_list), np.concatenate(conf_list)

    return C, R, L, conf

def main(im_name, alpha_list = [4.75, 5, 5.25, 5.5]):
    C_list, R_list, L_list, conf_list = [], [], [], []
    base = alpha_list[0]
    for alpha in tqdm(alpha_list):
        C, R, L, conf = predict(im_name, alpha=alpha)
        z = base / alpha
        C_list.append(np.round(np.array(C) * z).astype(int))
        R_list.append(np.round(np.array(R) * z).astype(int))
        L_list.append(L)
        conf_list.append(conf)
    C, R, L, conf = merge(C_list, R_list, L_list, conf_list)
    draw_circles(im_name, base, C, R, L, conf, out_name=f'outh.jpg')


main(im_name)
