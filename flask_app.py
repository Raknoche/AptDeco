from flask import Flask, flash, redirect, render_template, request, url_for, jsonify, session
import os
from random import randint
from werkzeug.utils import secure_filename
from ImageFeatures import *
import pickle
import pandas as pd
import cv2

app = Flask(__name__)

app.secret_key = 'super secret key'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_USER_IMAGES = os.path.join(APP_ROOT, 'static/userImages')
ALLOWED_EXTENSIONS = ["jpg", "png", "gif", "jpeg"]
user_images = []
image_probs = []


#Load the classifier
with open('image_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('image_classifier_features.pkl', 'rb') as f:
    image_classifier_features = pickle.load(f)
with open('classifier_mu_std.pkl', 'rb') as f:
    classifier_mu_std = pickle.load(f)
    mu = classifier_mu_std[0]
    std = classifier_mu_std[1]

#Handles getting skills from user PDF
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

'''Home Page Code'''
#Home page
@app.route("/", methods=["GET", "POST"])
def index():

    best_image_flag = 0
    best_image = []
    best_image_prob = []
    print("IMAGES:")
    print(user_images)

    if user_images:
        max_prob_idx = image_probs.index(max(image_probs))
        best_image = [user_images[max_prob_idx]]
        best_image_prob = [image_probs[max_prob_idx]]
        best_image_flag = 1

    return render_template("index.html", user_images=zip(user_images,image_probs),best_image_flag = best_image_flag, best_image=zip(best_image,best_image_prob))

@app.route('/get_images', methods=['POST'])
def get_images():
    # check if the post request has the file part
    file = request.files['file']
    
    if file and allowed_file(file.filename):

        #Save the Image
        filename = file.filename.rsplit('.', 1)[0].lower() + str(randint(0,100000)) + '.png'
        filename = secure_filename(filename)
        file_save_path = os.path.join(APP_USER_IMAGES, filename)
        file.save(file_save_path)

        user_images.append(filename)

        #Rate the image using our model
        image=cv2.imread(file_save_path)
        feats = ExtractFeatures(image)
        feats = pd.DataFrame(feats,index=[0])
        front_page_prob = clf.predict_proba(feats[image_classifier_features])[0][1]
        image_probs.append(front_page_prob)

    #Reload homepage
    return redirect(url_for('index'))

@app.route('/rank_images', methods=['GET','POST'])
def rank_images():
    if user_images:
        sorted_probs = sorted(image_probs,reverse=True)
        sorted_images = [img for (prob,img) in sorted(zip(image_probs,user_images),reverse=True)]

        sorted_stars = []
        for prob in sorted_probs:
            stars = (prob-mu)/(std/1.5)+3
            if stars <0:
                stars = 0
            if stars >5:
                stars = 5
            sorted_stars.append(stars)

        #max_prob_idx = image_probs.index(max(image_probs))
        #best_image = [user_images[max_prob_idx]]
        #best_image_prob = [image_probs[max_prob_idx]]
        best_image_flag = 1

    #Reload homepage
    return render_template("index.html", user_images=zip(sorted_images,sorted_stars),best_image_flag = best_image_flag)



if __name__ == "__main__":
    app.run()