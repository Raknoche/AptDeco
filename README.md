# DecoRater

My name is Richard Knoche, and I created DecoRater over three weeks while working at Insight Data Science. DecoRater was created for the peer-to-peer furniture resale website, AptDeco.com. The editors at AptDeco spend a large amount of time sifting through user uploaded images to determine which ones should be displayed on the front page of each listing. DecoRater automatically assesses the quality of each image, allowing editors to reduce the time spent looking at low quality images by 40%, and saving an estimated $45,000 on annual image curating and targeted advertising costs.

I was initially drawn to this project by the abstract nature of the problem. While there are some aspects of images which are easily quantified, the overall quality of an image is inherently subjective. I worked closely with the AptDeco team to understand what a “high quality” image meant to their editing team, and engineered over 60 features to treat these subjective opinions in a quantitative way.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

You'll need to install the following python packages to use the DecoRater codebase:

```
Python 3.5.0
conda install opencv
conda install flask
pip install imutils
pip install pycircstat
pip install decorater
pip install scipy
pip install nose
pip install pandas
pip install scikit-image
pip install scikit-learn=0.17.1
pip install matplotlib
pip install uuid
pip install pymysql
```

### Usage

ImageFeatures.py contains the bulk of the DecoRater code base.  The majority of functions within ImageFeatures.py are used to extract quantitative features from an image.  You can import the entire library to a python session using

```
from ImageFeatures import *
```

To classify an image from a url link, run the following code:


```
from ImageFeatures import *

img_url = '#PLACE_URL_HERE'
image   = UrlToImage(img_url)
image   = CropImage(image)

is_high_quality, high_quality_prob = ClassifyImage(image)
```

The first return of `ClassifyImage` will be `True` if the image is classified as high quality, and `False` otherwise.  The second return of `ClassifyImage` will return a floating point probability that the image is a high quality image.

The classifier is stored in `image_classifer.pkl`, and a list of features that it uses is stored in `image_classifier_features.pkl`.  

The code located in the `DecoRater` folder is used to run a flask web app.  To run your own web app, edit the first line of `run_local.py` to include your python path.  You can find your python path by typing `which python` in a command line terminal.  You will also need to create a local MySQL database to run the web app.  To do so, install MySQL, and run the following commands on your MySQL server:

```
create database DecoRaterUsers;
use DecoRaterUsers;
create table Sessions (user_id VARCHAR(100), user_images VARCHAR(3000), image_probs VARCHAR(3000));
```
After creating the database, edit the `DecoRater/static/sql_info` file to include your MySQL credentials.  Once complete, you can start the web server by running `python run_local.py` from the `AptDeco` directory.  The default url for the webserver is [http://localhost:5000](http://localhost:5000).

## Built With

* Front End: Flask, NGINX, Gunicorn, Bootstrap, AWS
* Back End: Python, MySQL, OpenCV

## Versioning

All version control is done with [Github](https://github.com/Raknoche/AptDeco). 

## Authors

**Richard Knoche**:

* [LinkedIn](https://www.linkedin.com/in/richardknoche)
* [Github](https://github.com/raknoche)
* [Data Science Blog](http://www.dealingdata.net/)

## Acknowledgments

* Ramon Cacho, for his expert insight on AptDeco's product
* Insight Data Science mentors and fellows, for their feedback on the project