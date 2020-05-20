from flask import Flask,render_template,Response,request,url_for,redirect
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import threading
import cvlib as cv
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.backend import set_session
import numpy as np
import time



app = Flask(__name__)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

# load the model, and pass in the custom metric function
global graph,sess

sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
model = load_model(r'model\trained_model.h5', custom_objects={'auc': auc})




@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        cap.open(0)
        return redirect(url_for('detector'))
    return render_template('home.html')

cap=cv2.VideoCapture(0)
time.sleep(2.0)

def video():
    
    while True:
        ret,frame=cap.read()
        with graph.as_default():
            set_session(sess)
            faces, confidences = cv.detect_face(frame) 
        
# loop over the face detections  
            if faces:
                for f in faces:
                    startX,startY,endX,endY=f

                    face = frame[startY:endY,startX:endX]
                    face=cv2.resize(face,(224,224))
                    face=img_to_array(face)
                    face=preprocess_input(face)
                    face=np.expand_dims(face,axis=0)
                    cv2.rectangle(frame,(startX,startY),(endX,endY),(255,0,0),2)
                    # with graph.as_default():
                    model._make_predict_function()
                    (mask,noMask)=model.predict(face)[0]
                    label="Mask" if mask>noMask else "No Mask"
                    colour=(0,255,0) if label=="Mask" else (0,0,255)

                    cv2.rectangle(frame, (startX,startY), (endX, endY), colour, 2)
                    cv2.putText(frame, label, (startX - 10, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
            if frame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
            # yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
    

@app.route('/mask_detect',methods=['GET','POST'])
def detector():
    if request.method=='POST':
        cap.release()
        return redirect(url_for('home'))
    return render_template('mask.html')

@app.route("/video_feed")
def video_feed():
    
	# return the response generated along with the specific media
	# type (mime type)
    return Response(video(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

with app.app_context():
    app.run(debug=False)

cap.release()