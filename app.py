#from jinja2 import Template
from tensorflow.python.keras.saving.saved_model.load import load
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf
#from tensorflow.keras.preprocessing import image
import numpy as np
import os
import keras
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from logging import debug
import login
import re

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


app = Flask(__name__)


img_size = (299,299)
preprocess_input = keras.applications.xception.preprocess_input
last_conv_layer_name = "block14_sepconv2_act"

# Loading our trained model for Atelectasis
atelectasis_model = load_model('static/all models/Atelectasis.h5' , compile=True)
cardiomegaly_model = load_model('static/all models/Cardiomegaly.h5', compile=True)
consolidation_model = load_model('static/all models/Consolidation.h5' , compile=True)
edema_model = load_model('static/all models/Edema.h5' , compile=True)
effusion_model = load_model('static/all models/Effusion.h5' , compile=True)
emphysema_model = load_model('static/all models/Emphysema.h5' , compile=True) 
fibrosis_model= load_model('static/all models/Fibrosis.h5',compile=True)
hernia_model = load_model('static/all models/Hernia.h5',compile=True)
infiltration_model = load_model('static/all models/Infiltration.h5' , compile=True)
lungopacity_model = load_model('static/all models/lung_opacity.h5' , compile=True)


# picfolder = os.path.join('static','upload')
# app.config['UPLOAD_FOLDER'] = picfolder

picfolder1 = os.path.join('static','uploadseparate')
#app.config['SEPARATE_UPLOAD_FOLDER'] = picfolder1

# path = "static/upload"
path1 = "static/uploadseparate"
# start of gradcam


def model_predict_for_disease(img_path, model):
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    preds = model.predict(img_array)
    return preds

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

#endof gradcam
def save_and_display_gradcam(img_path, heatmap, cam_path="static/gradcam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    #display(Image(cam_path))


def genrateSendOTP(email_id):
    import random
    import smtplib

    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login('#######','########')
    genrateSendOTP.otp = ''.join([str(random.randint(0,9)) for i in range(4)])
    msg='Hello, Your OTP is '+genrateSendOTP.otp
    print(msg)
    genrateSendOTP.otp = int(genrateSendOTP.otp)
    server.sendmail('###############',email_id,msg)
    server.quit()

def checkOTP(myotp):
    myotp = int(myotp)
    if genrateSendOTP.otp == myotp:
        return True
    else:
        return False


@app.route('/', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email_id = str(request.form['email'])

        import login
        login.genrateSendOTP(email_id)
        return render_template('otp.html',mail=email_id)
    else:
        return render_template('login.html')


@app.route('/loginCheck', methods=['GET','POST'])
def loginCheck():
    if request.method == 'POST':
        myotp = str(request.form['otp'])
        import login
        if login.checkOTP(myotp):
            return render_template('index.html')
        else:
            return render_template('login.html',msg="Entered Wrong OTP")



# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/affectedarea')
def aarea():
    return render_template('aarea.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/aboutus')
def About():
    return render_template('aboutus.html')


@app.route('/dev')
def dev():
    return render_template('devTeam.html')


@app.route('/upload')
def upload():
    return render_template('separatetest.html')


@app.route('/separatetest')
def separatetest():
    return render_template('separatetest.html')







# @app.route('/uploadingimage' ,  methods=['GET','POST'])
# def uploadingimage():
#     if request.method == 'POST':
#         f = request.files['chest-x-ray']
#         file1_path = os.path.join(path,secure_filename(f.filename))
#         f.save(file1_path)
#         pic1 = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
#         # Make Prediction
#         atelectasis_result = model_predict_for_atelactasis(file1_path, atelectasis_model)
#         return render_template('upload.html' , atelectasis_result = atelectasis_result
         
#          )

@app.route('/separateupload' ,  methods=['GET','POST'])
def separateupload():
    final_result = ""
    disease_name = ""
    probabilityofdisease = 0
    if request.method == 'POST':
        selectopt = request.form.get('selectopt')
        f = request.files['chest-x-ray']
        file1_path = os.path.join(path1,secure_filename(f.filename))
        f.save(file1_path)
        if selectopt == '1':
            disease_name = "Atelectasis"
            #Make Prediction
            atelectasis_result = model_predict_for_disease(file1_path, atelectasis_model)
            i = np.argmax(atelectasis_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = atelectasis_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = atelectasis_result[0][1]*100
                        # Prepare image
            
            img_path=file1_path
            print(img_path)
            img_array = preprocess_input(get_img_array(img_path, size=img_size))
            atelectasis_model.layers[-1].activation = None
            # Generate class activation heatmap
            # if u want to find gradient for specific class then choose value of i pr predictive index by yourself
            heatmap = make_gradcam_heatmap(img_array, atelectasis_model, last_conv_layer_name,i)
            
            save_and_display_gradcam(img_path, heatmap)
            gradcampath = "static/gradcam.jpg"
            print("hello-world")


        elif selectopt == '2':
            disease_name = "Cardiomegaly"
            #Make Prediction
            cardiomegaly_result = model_predict_for_disease(file1_path, cardiomegaly_model)
            i = np.argmax(cardiomegaly_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = cardiomegaly_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = cardiomegaly_result[0][1]*100
            img_path=file1_path
            print(img_path)
            img_array = preprocess_input(get_img_array(img_path, size=img_size))
            cardiomegaly_model.layers[-1].activation = None
            # Generate class activation heatmap
            # if u want to find gradient for specific class then choose value of i pr predictive index by yourself
            heatmap = make_gradcam_heatmap(img_array, cardiomegaly_model, last_conv_layer_name,i)
            
            save_and_display_gradcam(img_path, heatmap)
            gradcampath = "static/gradcam.jpg"
            print("hello-world")
        
        elif selectopt == '3':
            disease_name = "Consolidation"
            #Make Prediction
            consolidation_result = model_predict_for_disease(file1_path, consolidation_model)
            i = np.argmax(consolidation_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = consolidation_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = consolidation_result[0][1]*100
            img_path=file1_path
            print(img_path)
            img_array = preprocess_input(get_img_array(img_path, size=img_size))
            consolidation_model.layers[-1].activation = None
            # Generate class activation heatmap
            # if u want to find gradient for specific class then choose value of i pr predictive index by yourself
            heatmap = make_gradcam_heatmap(img_array, consolidation_model, last_conv_layer_name,i)
            
            save_and_display_gradcam(img_path, heatmap)
            gradcampath = "static/gradcam.jpg"
        
        elif selectopt == '4':
            disease_name = "Edema"
            #Make Prediction
            edema_result = model_predict_for_disease(file1_path, edema_model)
            i = np.argmax(edema_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = edema_result[0][0]*100
            else:
                final_result = "Normal"
                probabilityofdisease = edema_result[0][1]*100
            img_path=file1_path
            print(img_path)
            img_array = preprocess_input(get_img_array(img_path, size=img_size))
            edema_model.layers[-1].activation = None
            # Generate class activation heatmap
            # if u want to find gradient for specific class then choose value of i pr predictive index by yourself
            heatmap = make_gradcam_heatmap(img_array, edema_model, last_conv_layer_name,i)
            
            save_and_display_gradcam(img_path, heatmap)
            gradcampath = "static/gradcam.jpg"
        
        elif selectopt == '5':
            disease_name = "Effusion"
            #Make Prediction
            effusion_result = model_predict_for_disease(file1_path, effusion_model)
            i = np.argmax(effusion_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = effusion_result[0][0]*100
            else:
                final_result = "Normal"
                probabilityofdisease = effusion_result[0][1]*100
            img_path=file1_path
            print(img_path)
            img_array = preprocess_input(get_img_array(img_path, size=img_size))
            effusion_model.layers[-1].activation = None
            # Generate class activation heatmap
            # if u want to find gradient for specific class then choose value of i pr predictive index by yourself
            heatmap = make_gradcam_heatmap(img_array, effusion_model, last_conv_layer_name,i)
            
            save_and_display_gradcam(img_path, heatmap)
            gradcampath = "static/gradcam.jpg"
        
        elif selectopt == '6':
            disease_name = "Emphysema"
            #Make Prediction
            emphysema_result = model_predict_for_disease(file1_path, emphysema_model)
            i = np.argmax(emphysema_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = emphysema_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = emphysema_result[0][1]*100
            img_path=file1_path
            print(img_path)
            img_array = preprocess_input(get_img_array(img_path, size=img_size))
            emphysema_model.layers[-1].activation = None
            # Generate class activation heatmap
            # if u want to find gradient for specific class then choose value of i pr predictive index by yourself
            heatmap = make_gradcam_heatmap(img_array, emphysema_model, last_conv_layer_name,i)
            
            save_and_display_gradcam(img_path, heatmap)
            gradcampath = "static/gradcam.jpg"
        
        elif selectopt == '7':
            disease_name = "Fibrosis"
            #Make Prediction
            fibrosis_result = model_predict_for_disease(file1_path, fibrosis_model)
            i = np.argmax(fibrosis_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = fibrosis_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = fibrosis_result[0][1]*100
            img_path=file1_path
            print(img_path)
            img_array = preprocess_input(get_img_array(img_path, size=img_size))
            fibrosis_model.layers[-1].activation = None
            # Generate class activation heatmap
            # if u want to find gradient for specific class then choose value of i pr predictive index by yourself
            heatmap = make_gradcam_heatmap(img_array, fibrosis_model, last_conv_layer_name,i)
            
            save_and_display_gradcam(img_path, heatmap)
            gradcampath = "static/gradcam.jpg"
        
        elif selectopt == '8':
            disease_name = "Hernia"
            #Make Prediction
            hernia_result = model_predict_for_disease(file1_path, hernia_model)
            i = np.argmax(hernia_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = hernia_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = hernia_result[0][1]*100
            img_path=file1_path
            print(img_path)
            img_array = preprocess_input(get_img_array(img_path, size=img_size))
            hernia_model.layers[-1].activation = None
            # Generate class activation heatmap
            # if u want to find gradient for specific class then choose value of i pr predictive index by yourself
            heatmap = make_gradcam_heatmap(img_array, hernia_model, last_conv_layer_name,i)
            
            save_and_display_gradcam(img_path, heatmap)
            gradcampath = "static/gradcam.jpg"
        
        elif selectopt == '9':
            disease_name = "Infiltration"
            #Make Prediction
            infiltration_result = model_predict_for_disease(file1_path, infiltration_model)
            i = np.argmax(infiltration_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = infiltration_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = infiltration_result[0][1]*100
            img_path=file1_path
            print(img_path)
            img_array = preprocess_input(get_img_array(img_path, size=img_size))
            infiltration_model.layers[-1].activation = None
            # Generate class activation heatmap
            # if u want to find gradient for specific class then choose value of i pr predictive index by yourself
            heatmap = make_gradcam_heatmap(img_array, infiltration_model, last_conv_layer_name,i)
            
            save_and_display_gradcam(img_path, heatmap)
            gradcampath = "static/gradcam.jpg"
        
        elif selectopt == '10':
            disease_name = "Lung Opacity"
            #Make Prediction
            lungopacity_result = model_predict_for_disease(file1_path, lungopacity_model)
            i = np.argmax(lungopacity_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = lungopacity_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = lungopacity_result[0][1]*100
            img_path=file1_path
            print(img_path)
            img_array = preprocess_input(get_img_array(img_path, size=img_size))
            lungopacity_model.layers[-1].activation = None
            # Generate class activation heatmap
            # if u want to find gradient for specific class then choose value of i pr predictive index by yourself
            heatmap = make_gradcam_heatmap(img_array, lungopacity_model, last_conv_layer_name,i)
            
            save_and_display_gradcam(img_path, heatmap)


    return render_template('separatetest.html' , final_result = final_result , probabilityofdisease = probabilityofdisease ,
    disease_name = disease_name , gradcampath="static/gradcam.jpg")


if __name__ == '__main__':
    app.run(debug=True)
