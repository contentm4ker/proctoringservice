import numpy as np
import cv2
from yolov3 import YoloV3, load_darknet_weights

PERSON_CLASS = 0
MOBILE_PHONE_CLASS = 67
    
def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]), x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img

def preprocess_image_for_yolo(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    img = img / 255
    return img

if __name__ == '__main__':
    CLASS_NAMES = [c.strip() for c in open('../models/classes.txt').readlines()]
    yolo = YoloV3()
    load_darknet_weights(yolo, '../models/yolov3.weights')

    cap = cv2.VideoCapture(0)

    while(True):
        ret, image = cap.read()
        if ret == False:
            break
        img = preprocess_image_for_yolo(image)
        boxes, scores, classes, nums = yolo(img)
        count = 0
        for i in range(nums[0]):
            if int(classes[0][i] == PERSON_CLASS):
                count += 1
            if int(classes[0][i] == MOBILE_PHONE_CLASS):
                print('mobile phone detected')
        if count == 0:
            print('no person')
        elif count > 1: 
            print('more than one person')
            
        image = draw_outputs(image, (boxes, scores, classes, nums), CLASS_NAMES)

        cv2.imshow('prediction', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

