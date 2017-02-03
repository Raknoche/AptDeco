from DecoRater import app
from flask import Flask, flash, redirect, render_template, request, url_for, jsonify
import os
from random import randint
from werkzeug.utils import secure_filename
from ImageFeatures import *
import pickle
import cv2
import pandas as pd


#Set up application parameters
app.secret_key = 'some_secret'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_USER_IMAGES = os.path.join(APP_ROOT, 'static/userImages')
ALLOWED_EXTENSIONS = ["jpg", "png", "gif", "jpeg"]

#Global lists -- need to change this to SQL database
user_images = []
image_probs = []


#Load the classifier
with open('image_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('image_classifier_features.pkl', 'rb') as f:
    image_classifier_features = pickle.load(f)


#Refuses file extentions that aren't in ALLOWED _EXTENSIONS
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Home page
@app.route("/", methods=["GET", "POST"])
def index():

    #Clear all user lists on page refresh
    del user_images[:]
    del image_probs[:]

    return render_template("index.html")

#About page
@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

#Google Slides
@app.route("/slides")
def slides():
    ''' Show slides '''
    return render_template("slides.html")

#Handles dropzone uploads
@app.route('/get_images', methods=['POST'])
def get_images():
    file = request.files['file']
    if file and allowed_file(file.filename):

        #Save the Image
        filename = file.filename.rsplit('.', 1)[0].lower() + str(randint(0,100000)) + '.png'
        filename = secure_filename(filename)
        file_save_path = os.path.join(APP_USER_IMAGES, filename)
        file.save(file_save_path)

        #Rate the image using our model
        image=cv2.imread(file_save_path)
        feats = ExtractFeatures(image)
        feats = pd.DataFrame(feats,index=[0])
        front_page_prob = clf.predict_proba(feats[image_classifier_features])[0][1]

        #Append global lists
        user_images.append(filename)
        image_probs.append(front_page_prob)

    #Return nothing
    return ('',204)


#Ranked images page
@app.route('/rank_images', methods=['GET','POST'])
def rank_images():

    #Created sorted lists of the user's images
    sorted_images=[]
    sorted_stars=[]
    sorted_probs=[]
    if user_images:
        for (prob,img) in sorted(zip(image_probs,user_images),reverse=True):
            if img not in sorted_images:
                sorted_images.append(img)
                sorted_probs.append(prob)

                stars = prob*5
                sorted_stars.append(stars)

        return render_template("ranked_images.html", user_images=list(zip(sorted_images,sorted_stars)) )
    else:
        flash('Please upload at least one jpg, png, or gif file before submitting.')
        return redirect(url_for('index'))


if __name__ == "__main__":
    app.run()
