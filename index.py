from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

def highlightFace(net, frame, conf_threshold=0.7):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
    return faceBoxes



@app.route('/detect', methods=['POST'])
def detect_age_gender():
    # Check if the request contains an image file
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    if "distUID" not in request.form:
        return jsonify({"error": "No distUID provided"})

    # Read the image file from the request
    image = request.files['image'].read()

    # Convert the image data to OpenCV format
    nparr = np.fromstring(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # Load the pre-trained age and gender detection model
    faceProto="opencv_face_detector.pbtxt"
    faceModel="opencv_face_detector_uint8.pb"
    ageProto="age_deploy.prototxt"
    ageModel="age_net.caffemodel"
    genderProto="gender_deploy.prototxt"
    genderModel="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    # ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    ageList=[1, 5, 10, 17, 28, 40, 50, 80]
    genderList=['H','F']
    padding=20

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)
    # Convert the image to grayscale for face detection
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image

    faceBoxes = highlightFace(faceNet,img)

    if not faceBoxes:
        return jsonify({"message": "No faces detected"}), 400
    
    detections = []

    for faceBox in faceBoxes:
        face=img[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,img.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, img.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        # print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        # print(f'Age: {age[1:-1]} years')

        detections.append({'age' : age, 'gender' : gender})
    backend = os.environ.get('backend')
    url = backend + '/advertisement/getAdByAgeGender'
    headers = {'Content-Type': 'application/json'}
    data = {'distUID' : request.form['distUID'], 'age': detections[0]['age'], 'sexe': detections[0]['gender']}

    response = requests.get(url, json=data, headers=headers)

    if response.status_code == 200:
        return {"statusCode": 200,
                "status": "Succes",
                "message": "Succes",
                "data": response.json().get('data')}, 200
    else:
        return {"statusCode": 500,
                "status": "Failure",
                "message": "Internal Error",
                "data": None}, 500


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=80)