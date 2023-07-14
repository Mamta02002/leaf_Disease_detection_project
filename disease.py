from flask import Flask, render_template, request, url_for, redirect

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import cv2

app=Flask(__name__)
@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/analyze",methods=["GET","POST"])
def analyze():
    return render_template('analyze.html')

@app.route("/predict", methods = ["GET","POST"])
def predict():
    if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename 
        
        file_path = os.path.join('static/upload/', filename)
        file.save(file_path)
        file.close()
    model = load_model("leaf_classifier_model.h5")
    def prediction(path):
        img = tf.keras.preprocessing.image.load_img(path,target_size=(256,256))
        i = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(i,axis=0)
        pred = np.argmax(model.predict(img))
        classes = {'apple': 0, 'potato': 1}
        return list(classes.keys())[list(classes.values()).index(pred)]

    # def severity_prediction(path):
    #     leaf = cv2.imread(path)
    #     img = cv2.cvtColor(leaf,cv2.COLOR_BGR2HSV)
    #     low_full = np.array([0,0,0])
    #     high_full = np.array([255,255,255])
    #     x_full = cv2.inRange(img,low_full, high_full)
    #     low_leaf = np.array([0,0,0])
    #     high_leaf = np.array([120,250,255])
    #     x_leaf = cv2.inRange(img,low_leaf, high_leaf)
    #     back_low = np.array([0,0,40])
    #     back_high = np.array([180,40,230])
    #     x = cv2.inRange(img,back_low, back_high)
    #     low_green = np.array([35,40,20])
    #     high_green = np.array([85,255,255])
    #     x_healthy = cv2.inRange(img,low_green, high_green)
    #     x_back = x_full-x_leaf
    #     x_disease = x-x_back
    #     x_leaf = x_disease+x_healthy
    #     return np.count_nonzero(x_disease)/np.count_nonzero(x_leaf)
    #     # print(np.count_nonzero(x_healthy)/np.count_nonzero(x_leaf)) healthy part %

    leaf = prediction(file_path)
    if leaf=='potato':
        model = load_model('potatoes.h5')
        image = tf.keras.preprocessing.image.load_img(file_path,target_size=(256,256))
        x = tf.keras.preprocessing.image.img_to_array(image)
        img_test = np.expand_dims(x, axis=0)
        class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        batch_prediction=model.predict(img_test)
        result=class_names[np.argmax(batch_prediction[0])]
        return render_template('predict.html',prediction=result,user_image=file_path)
    
    # elif leaf=='tomato':
    #     model = load_model('tomatoes.h5')
    #     image = tf.keras.preprocessing.image.load_img(file_path,target_size=(256,256))
    #     x = tf.keras.preprocessing.image.img_to_array(image)
    #     img_test = np.expand_dims(x, axis=0)
    #     class_names=['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
    #                  'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    #                  'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
    #     batch_prediction=model.predict(img_test)
    #     result=class_names[np.argmax(batch_prediction[0])]
    #     sev = f"{severity_prediction(file_path)*100}%"
    #     return render_template('predict.html',prediction=result,severity = sev,user_image=file_path)
    else:
        model = load_model('apples.h5')
        image = tf.keras.preprocessing.image.load_img(file_path,target_size=(256,256))
        x = tf.keras.preprocessing.image.img_to_array(image)
        img_test = np.expand_dims(x, axis=0)
        class_names=['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']
        batch_prediction=model.predict(img_test)
        result=class_names[np.argmax(batch_prediction[0])]
        return render_template('predict.html',prediction=result,user_image=file_path)

@app.route("/predict_severity", methods = ["GET","POST"])
def predict_severity():
    if request.method == 'POST':
        file = request.files['image'] # fetch input
        filename = file.filename 
        file2 = request.files['annot']
        filename2 = file2.filename

        file_path = os.path.join('static/upload/', filename)
        annot_path = os.path.join('static/upload/',filename2)
        file.save(file_path)
        file2.save(annot_path)
    
    import apple
    s = apple.Severity()
    sev = float(s.calculate_leaf_disease_severity(file_path,annot_path))
    bbpath = s.create_bounding_box(file_path,annot_path,sev)
    return render_template('predict.html',severity = f"{sev}%",user_image=bbpath)

@app.route("/apple")
def apple():
    return render_template('apple.html')

# @app.route("/home")
# def home():
#     return render_template('home.html')

# @app.route("/about")
# def about():
#     return render_template('about.html')
# @app.route("/apple")
# def contact():
#     return render_template('contact.html')
if __name__=="__main__":
    app.run(debug=True)
