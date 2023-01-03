import cv2
import torch
from PIL import Image
import os, sys
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    filenames = []
    for boxes in root.iter('object'):

        filename = root.find('filename').text
        filenames.append(filename)
        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filenames, list_with_all_boxes

name, boxes = read_content("file.xml")
print(name)
print(boxes)
# Model
model = torch.hub.load('yolov5', 'custom', path='yolov5/best.pt', source='local')  # local repo

# Images
OpenedImg = []
predict_datas = []
for imgName, box in zip(name, boxes):
    print("imgName is " + imgName)
    Img = Image.open('testImg/' + imgName)
    #new_img = img.crop((0, 0, 256, 512))  # (left, upper, right, lower)
    print(imgName)
    #               ymin:ymax , xmin:xmax
    crop_img =Img.crop( box   )
    result = model( crop_img )
    print("class for img " +imgName+" :")
    print(result.pandas().xyxy[0]["name"][0])
    predict_data = [ imgName,result.pandas().xyxy[0]["name"][0] ]
    predict_datas.append(predict_data)
with open('group24.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(predict_datas)
# Inference
#print(results.pandas().xyxy)
# Results
#results.print()  
#results.save()  # or .show()

#results.xyxy[0]  # im1 predictions (tensor)

#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie