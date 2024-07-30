







# import cv2
# import numpy as np
# import argparse

# argparser = argparse.ArgumentParser(description="")

# argparser.add_argument('--img',type=str)
# argparser.add_argument('--video',type=str)
# argparser.add_argument('--out',type=str)

# def predect(net,layer_names,hight,width,img,labels):
#     blob = cv2.dnn.blobFromImage(cv2.resize(img, 1/255.0,(416,416), swapRB = True,crop=False))
#     net.setInput(blob)
#     outs = net.forward(layer_names)

#     boxes , confidences , class_ids = [],[],[]

#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#     idxs = cv2.dnn.NMSBoxes(boxes,confidences, confidence )

#     img = draw_bb(img ,boxes,confidences,class_ids,idxs,labels)

#     return img , boxes,confidences,class_ids,idxs

import cv2
import argparse
import numpy as np

argparser = argparse.ArgumentParser(description="Object Detection Script")
argparser.add_argument('--img', type=str)
argparser.add_argument('--video', type=str)
argparser.add_argument('--out', type=str)

def draw_bb(img, boxes, confidences, class_ids, idxs, labels):
    # Dummy function for illustration
    return img

def predict(net, layer_names, height, width, img, labels):
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layer_names)

    # init lists of detected boxes, confidences, class IDs
    boxes, confidences, class_ids = [], [], []

    # loop over each of the layer outputs
    for out in layerOutputs:
        # loop over each of the detections
        for detection in out:
            # extract the class ID and confidence of the current OD
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Suppress overlapping boxes
            if confidence > 0.5:  # Example confidence threshold
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Example thresholds

    img = draw_bb(img, boxes, confidences, class_ids, idxs, labels)

    return img, boxes, confidences, class_ids, idxs


















# this file is note work than comment on this repo and i will arenge new and updeted code for you