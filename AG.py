# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 00:48:19 2020

@author: Tulasi
"""


# USAGE

import cv2 as cv #opencv
import time #time
from flask import Flask,request, render_template
import os
from werkzeug.utils import secure_filename

#face Detection
faceProto = "opencv_face_detector.pbtxt" 
faceModel = "opencv_face_detector_uint8.pb"

#Age Predition
ageProto = "age_deploy.prototxt" #weight file training data
ageModel = "age_net.caffemodel" #model file

#Gender Detetion
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'] #age list
genderList = ['Male', 'Female'] # Gender List

# Load network
ageNet = cv.dnn.readNetFromCaffe(ageProto,ageModel)#Age #dnn-deep neural network is a pre trained model
genderNet = cv.dnn.readNetFromCaffe(genderProto,genderModel)#Gender
faceNet = cv.dnn.readNet(faceModel,faceProto)#Face


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()#stores the face data
    bboxes = []
    for i in range(detections.shape[2]): #drawing the rectangles
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes
    
app=Flask(__name__,template_folder="templates")
@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')
@app.route('/home', methods=['GET'])
def about():
    return render_template('home.html')
@app.route('/image1',methods=['GET','POST'])
def image1():
    return render_template("index6.html")

@app.route('/predict',methods=['GET','POST'])
def image():
    if request.method == 'POST':
        print("inside image")
        f = request.files['image']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)   
        print(file_path)      
    cap = cv.VideoCapture(file_path)
    padding = 20
    while cv.waitKey(1) < 0:
    # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue

        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):
                         min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            label = "{},{}".format(gender, age)
            cv.putText(frameFace, label, (bbox[0]-5, bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255),
                       2, cv.LINE_AA)
            cv.imshow("Age Gender Demo", frameFace)
        if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
        # Release handle to the webcam
    cap.release()
    cv.destroyAllWindows()
    
    return render_template("index6.html")

@app.route('/upload', methods=['GET', 'POST'])
def predict():
   
        # Load images.
    cap = cv.VideoCapture(0)
    padding = 20
    while cv.waitKey(1) < 0:
    # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue

        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
        
       # print("Age : {}, confidence = {:.3f}".format(age, agePreds[0].max()))

            label = "{},{}".format(gender, age)
            cv.putText(frameFace, label, (bbox[0]-5, bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255), 2, cv.LINE_AA)
            cv.imshow("Age Gender Demo", frameFace)
            #name = args.i
        #cv.imwrite('./detected/'+name,frameFace)
        #print("Time : {:.3f}".format(time.time() - t))
        if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
        # Release handle to the webcam
    cap.release()
    cv.destroyAllWindows()
        
    return render_template("home.html")

    
if __name__ == '__main__':
      app.run(port=8000, debug=False)
 