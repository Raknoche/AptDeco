from DecoRater import app
from flask import Flask, flash, redirect, render_template, request, url_for, jsonify, session
import os
from random import randint
from werkzeug.utils import secure_filename
from ImageFeatures import *
import pickle
import cv2
import pandas as pd
import uuid
import pymysql as mdb

#Set up application parameters
app.secret_key = 'some_secret'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_USER_IMAGES = os.path.join(APP_ROOT, 'static/userImages')
APP_SQL_INFO = os.path.join(APP_ROOT, 'static/sql_info')

#Get SQL DB info
with open(APP_SQL_INFO) as f:
    sql_content = f.readlines()
for var in sql_content:
    exec(var)

#Load the classifier
with open('image_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('image_classifier_features.pkl', 'rb') as f:
    image_classifier_features = pickle.load(f)

#Home page
@app.route("/", methods=["GET", "POST"])
def index():
    if(not 'uid' in session):
        session['uid'] = uuid.uuid4()

    #Clear all images for this session
    con = mdb.connect(sql_address, sql_user, sql_password, sql_database, charset='utf8');
    with con:
        cur = con.cursor()
        query = "DELETE FROM Sessions WHERE user_id = %s"
        cur.execute(query,(str(session['uid'])))

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
    #Dropbox JS will handle extension restrictions
    file = request.files['file']

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

    #Inserting into Session DB
    con = mdb.connect(sql_address, sql_user, sql_password, sql_database, charset='utf8');
    with con:
        cur = con.cursor()
        query = "INSERT INTO Sessions (user_id,user_images,image_probs) VALUES (%s,%s,%s)"
        cur.execute(query,(str(session['uid']),filename,float(front_page_prob)))

    #Return nothing
    return ('',204)


#Ranked images page
@app.route('/rank_images', methods=['GET','POST'])
def rank_images():

    #Retrieving the user's images
    con = mdb.connect(sql_address, sql_user, sql_password, sql_database, charset='utf8');
    with con:
        cur = con.cursor()
        query = "SELECT user_images,image_probs FROM Sessions WHERE user_id = %s"
        cur.execute(query,(str(session['uid'])))
        res = cur.fetchall()

    user_images = [img for (img,prob) in res]
    image_probs = [np.float(prob) for (img,prob) in res]

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
