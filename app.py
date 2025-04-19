from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import model_from_json
# import time
from dotenv import load_dotenv
import os
import base64
from requests import post, get
import json

app = Flask(__name__)

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
print(client_id)
print(client_secret)


def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')

    url = "https://accounts.spotify.com/api/token"

    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "client_credentials",
    }

    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token


def get_auth_headers(token):
    return {"Authorization": "Bearer " + token}


def search_for_album(token, emotion):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_headers(token)
    query = f"?q={emotion}&type=playlist&limit=1"

    query_url = url + query
    result = get(query_url, headers=headers)
    print("inside search function")

    json_result = json.loads(result.content)['playlists']['items']

    if len(json_result) == 0:
        print("No artist found")
        return None
    return json_result[0]


def get_emo(emotion):
    token = get_token()
    result = search_for_album(token, emotion)
    print("inside get emo function")
    if result is not None:  # Check if result is None
        playlist_id = result['external_urls']
        print(playlist_id['spotify'])
    else:
        print(f"No playlist found for emotion: {emotion}")
        # Add a default playlist or handle the error as needed
        #For example:
        #print("Default playlist link")
        #print("default playlist link here")


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)


def gen_frames():
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))

        if not ret:
            break

        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            print("working 1")
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            print("working 2")
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            print("working 3")
            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
            curr_emotion = emotion_dict[maxindex]
            get_emo(curr_emotion)
            print("ran till face detection")

        # encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # yield the frame to the web page
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        break

message = "test"

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/playlist_link')
def playlist_link():
    return Response(message)


if __name__ == '__main__':
    app.run(debug=True)
