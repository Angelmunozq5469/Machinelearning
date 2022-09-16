import os
import string
import cv2
import numpy as pd
import csv

pneumonia = "PNEUMONIA"
normalpath = "NORMAL"
neumonia = os.listdir(pneumonia)
normal = os.listdir(normalpath)
features = []
chart = []
for p in neumonia:
    imagenesIMGNEU = pneumonia + "/" + p
    imagepne = cv2.imread(imagenesIMGNEU)
    imagepne = cv2.resize(imagepne, [500, 500])
    imagepne = cv2.cvtColor(imagepne, cv2.COLOR_BGR2GRAY)   
    humm = cv2.HuMoments(cv2.moments(imagepne)).flatten()
    chart = [humm[0],humm[1],humm[2],humm[3],humm[4],humm[5],humm[6],1]
    features.append(chart)
    test = open("pneumonia.csv","w")
    wr = csv.writer(test, dialect="excel")
    for item in features:
        wr.writerow(item)

# cv2.destroyWindow

for n in normal:
    imageNormal = normalpath + "/" + n
    imageNor = cv2.imread (imageNormal)
    imageNor = cv2.resize(imageNor, [500, 500])
    imageNor = cv2.cvtColor(imageNor, cv2.COLOR_BGR2GRAY)      
    humm = cv2.HuMoments(cv2.moments(imageNor)).flatten()   
    chart = [humm[0],humm[1],humm[2],humm[3],humm[4],humm[5],humm[6],0]    
    features.append(chart)
    test = open("pneumonia.csv","w")
    wr = csv.writer(test, dialect="excel")
    for item in features:
        wr.writerow(item)

cv2.destroyWindow