import cv2
import imutils
import numpy as np
import pycircstat.descriptive as circstat
import pySaliencyMap as SMap
import itertools
import urllib
from skimage.filters.rank import entropy
from skimage.morphology import disk
import pickle
import pandas as pd

def ResizeImage(image):
    '''Resizes an image to 800x800 pixels. 
    This ensures feature values are produced on the same scale.

    Keyword Arguments:
    image -- The BGR image collected by UrlToImage()

    Output:
    scaled_image -- An 800x800 version of the BGR image.
    '''

    scaler = np.min([800.0/image.shape[0], 800.0/image.shape[1]])
    scaled_image = cv2.resize(image,(np.int(scaler*image.shape[1]),\
        np.int(scaler*image.shape[0])),interpolation=cv2.INTER_AREA)

    return scaled_image

def ExtractRGB(image):
    '''Extracts the RGB color space from an image.

    Keyword Arguments:
    image -- The BGR image collected by UrlToImage()

    Output:
    r -- The red channel of the image
    b -- The blue channel of the image
    g -- The green channel of the image
    '''

    b,g,r = cv2.split(image)

    return r, g, b

def ExtractHSV(image):
    '''Extracts The HSV color space from an image.

    Keyword Arguments:
    image -- The BGR image collected by UrlToImage()

    Output:
    h -- The hue channel of the image, in degrees
    s -- The saturation channel of the image
    v -- The value channel of the image
    h_rad --  The hue channel of the image, in radians
    '''

    hsv   = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)
    h_rad = h.flatten()*np.pi/180.0 #Hue converted to radians

    return h, s, v, h_rad

def ExtractGrayscale(image):
    '''Extracts the grayscale version of an image.

    Keyword Arguments:
    image -- The BGR image collected by UrlToImage()

    Output:
    gray -- The grayscale version of the image.
    '''

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray

def ExtractLAB(image):
    '''Extracts the LAB color space from an image.

    Keyword Arguments:
    image -- The BGR image collected by UrlToImage()

    Output:
    l -- The lightness channel of the image
    a -- The a color opponent channel of the image
    b -- The b color opponent channel of the image
    '''    

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab_image)

    return l, a, b


def MeasureWhite(image):
    '''Calculates the fraction of pixels in the image which are white.
    Useful for determining if an image is already edited or not.

    Keyword Arguments:
    image -- The BGR image collected by UrlToImage()

    Output:
    frac_white -- The fraction of pixels which are white.
    '''    

    white_pixels = (image == 255).all(axis=2)
    frac_white = len(white_pixels[white_pixels==True])/white_pixels.size

    return frac_white


def CropImage(image):
    '''Removes white borders from images. If frac_white > 50%, 
    we assume it is background subtracted image, so we don't crop it.

    Keyword Arguments:
    image -- The BGR image collected by UrlToImage()

    Output:
    image -- A version of the input image with white borders removed.
    '''  

    if MeasureWhite(image) < 0.50:
        img_g    = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        good_row = np.where(np.var(image,axis=1)>2)[0]
        good_col = np.where(np.var(image,axis=0)>10)[0]
        img2     = image[:,min(good_col):max(good_col)+1,:]
        image    = img2[min(good_row):max(good_row)+1,:,:]

    return image

def LaplacianSharpness(gray):
    '''Runs a laplacian filter over a grayscale image, and uses 
    the variance of the filter as a measure of the image sharpness.

    Keyword Arguments:
    gray -- The gray channel of an image

    Output:
    sharpness -- The sharpness of the image, ranging from 0 to infinity.  Higher
    values correspond to sharper images.

    Source:
    http://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    '''  

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    return sharpness

def FFTSharpness(gray):
    '''Calculates the sharpness of an image from the fourier transform.

    Keyword Arguments:
    gray -- The gray channel of an image

    Output:
    average_fft -- The average of the FFT spectrum.

    Source:
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html

    '''  

    rows, cols  = gray.shape
    crow, ccol  = int(rows/2),int(cols/2)
    f           = np.fft.fft2(gray)
    fshift      = np.fft.fftshift(f)
    fshift[crow-75:crow+75, ccol-75:ccol+75] = 0
    f_ishift    = np.fft.ifftshift(fshift)
    img_fft     = np.fft.ifft2(f_ishift)
    img_fft     = 20*np.log(np.abs(img_fft))
    average_fft = np.mean(img_fft)

    return average_fft

def Colorfulness(a,b):
    '''Calculates the colorfulness of an image using the methods from
    https://www.researchgate.net/publication/243135534_Measuring_Colourfulness_in_Natural_Images
    
    Keyword Arguments:
    a -- The a color opponent channel of an image
    b -- The b color opponent channel of an image

    Output:
    colorfulness -- The colorfulness of the image.  
                    Ranges from 0 to infinity, with higher values being more colorful.
    ''' 

    mu_ab = np.sqrt( a.mean()**2 + b.mean()**2)
    colorfulness = a.std() + b.std() + 0.39*mu_ab

    return colorfulness

def RGBStats(image):
    '''Calculates the mean and standard deviation of the RGB color channels.
    Input should be a BGR image format. Note that standard deviations are
    normalized to the mean of the color channel.

    Keyword Arguments:
    image -- The BGR image collected by UrlToImage()

    Output:
    r_mean -- The average value of the red channel
    g_mean -- The average value of the green channel
    b_mean -- The average value of the blue channel
    r_std  -- The standard deviation of the red channel, relative to the mean
    g_std  -- The standard deviation of the green channel, relative to the mean
    b_std  -- The standard deviation of the blue channel, relative to the mean
    ''' 

    (rgb_means, rgb_stds) = cv2.meanStdDev(image)
    b_mean,g_mean,r_mean  = rgb_means[:,0]
    b_std,g_std,r_std     = rgb_stds[:,0]/rgb_means[:,0]

    return r_mean, g_mean, b_mean, r_std, g_std, b_std

def RGBGradients(r,g,b):
    '''Calculates the mean and standard deviation of color gradients
    in each color channel.

    Keyword Arguments:
    r -- The red channel of the image.
    g -- The green channel of the image.
    b -- The blue channel of the image

    Output:
    r_xgrad      -- The average gradient of the red channel in the horizontal direction
    g_xgrad      -- The average gradient of the green channel in the horizontal direction
    b_xgrad      -- The average gradient of the blue channel in the horizontal direction
    r_ygrad      -- The average gradient of the red channel in the vertical direction
    g_ygrad      -- The average gradient of the green channel in the vertical direction
    b_ygrad      -- The average gradient of the blue channel in the vertical direction
    r_xgrad_std  -- The standard deviation of the red channel gradients in the horizontal direction
    g_xgrad_std  -- The standard deviation of the green channel gradients in the horizontal direction
    b_xgrad_std  -- The standard deviation of the blue channel gradients in the horizontal direction
    r_ygrad_std  -- The standard deviation of the red channel gradients in the vertical direction
    g_ygrad_std  -- The standard deviation of the green channel gradients in the vertical direction
    b_ygrad_std  -- The standard deviation of the blue channel gradients in the vertical direction
    '''

    r_xgrad     = np.mean(cv2.Sobel(r,cv2.CV_64F,1,0,ksize=1))
    g_xgrad     = np.mean(cv2.Sobel(g,cv2.CV_64F,1,0,ksize=1))
    b_xgrad     = np.mean(cv2.Sobel(b,cv2.CV_64F,1,0,ksize=1))

    r_ygrad     = np.mean(cv2.Sobel(r,cv2.CV_64F,0,1,ksize=1))
    g_ygrad     = np.mean(cv2.Sobel(g,cv2.CV_64F,0,1,ksize=1))
    b_ygrad     = np.mean(cv2.Sobel(b,cv2.CV_64F,0,1,ksize=1))

    r_xgrad_std = np.std(cv2.Sobel(r,cv2.CV_64F,1,0,ksize=1))
    g_xgrad_std = np.std(cv2.Sobel(g,cv2.CV_64F,1,0,ksize=1))
    b_xgrad_std = np.std(cv2.Sobel(b,cv2.CV_64F,1,0,ksize=1))

    r_ygrad_std = np.std(cv2.Sobel(r,cv2.CV_64F,0,1,ksize=1))
    g_ygrad_std = np.std(cv2.Sobel(g,cv2.CV_64F,0,1,ksize=1))
    b_ygrad_std = np.std(cv2.Sobel(b,cv2.CV_64F,0,1,ksize=1))

    return r_xgrad, g_xgrad, b_xgrad, r_ygrad, g_ygrad, b_ygrad, \
            r_xgrad_std, g_xgrad_std, b_xgrad_std, r_ygrad_std, g_ygrad_std, b_ygrad_std

def HSVStats(h_rad,s,v):
    '''Caclulates mean and standard deviation of the HSV color channels.
    We use circular stats for hue, since hue is defined over a unit cirlce
    and the average of an angle on a unit circle is not the same as numerical average.

    Keyword Arguments:
    h_rad -- The hue channel of an image, in radians
    s     -- The saturation channel of an image
    v     -- The value channel of an image

    Output:
    h_mean -- The average value of the hue channel
    s_mean -- The average value of the saturation channel
    v_mean -- The average value of the value channel
    h_std  -- The standard deviation of the hue channel
    s_std  -- The standard deviation of the saturation channel
    v_std  -- The standard deviation of the value channel
    '''

    h_mean = circstat.mean(h_rad)*180.0/np.pi           
    h_var  = circstat.var(h_rad)*180.0/np.pi

    s_mean = np.mean(s)/255.0                           
    s_var  = np.var(s/255.0)

    v_mean = np.mean(v)/255.0                           
    v_var  = np.var(v/255.0)

    return h_mean, s_mean, v_mean, h_var, s_var, v_var

def HSVBlur(h,s,v):
    '''Uses a laplacian filter to calculate the blurriness of the HSV
    image channels.

    Keyword Arguments:
    h  -- The hue channel of an image, in degrees
    s  -- The saturation channel of an image
    v  -- The value channel of an image

    Output:
    h_laplacian -- The variance of the laplacian of the Hue channel
    s_laplacian -- The variance of the laplacian of the Saturation channel
    v_laplacian -- The variance of the laplacian of the Value channel
    '''

    h_laplacian = cv2.Laplacian(h/255.0, cv2.CV_64F).var()
    s_laplacian = cv2.Laplacian(s/255.0, cv2.CV_64F).var()
    v_laplacian = cv2.Laplacian(v/255.0, cv2.CV_64F).var()

    return h_laplacian, s_laplacian, v_laplacian

def ComplimentaryColors(h_rad):
    '''Quantifies how complimentary the colors of an image are.
    Calculation uses the complex plan as a unit circle.  
    To compare how complimentary two colors are, we multiple the hue of the
    second color by two.  This way, exp(h*j) will fall at (1,0j) and (1,0j) for 
    colors opposite each other on the hue circle, and (1,0j) and (-1,0j) for
    colors at 90 degree separation on the circle.

    Keyword Arguments:
    h_rad  -- The hue channel of an image, in radians

    Output:
    color_compliment -- The level of complimentary colors in the image.
                        Ranges from 0 to 1, with 1 being completely complementary.
    '''

    color_compliment = np.abs(np.exp(2*h_rad*1j).sum() / len(h_rad.flatten()))

    return color_compliment


def HistogramDarkness(image):
    '''Calculates the average value of the color histogram
    as a mesurement for image darkness.

    Keyword Arguments:
    image -- The BGR image collected by UrlToImage()

    Output:
    hist_darkness -- The mean value of the image histogram.
                     Ranges from 0 to 255 between pure white and pure black images.
    '''

    hist,bins = np.histogram(image.ravel(),255,[0,255])
    bin_centers = (bins[1:]+bins[:-1])/2
    hist_darkness = (bin_centers*hist).sum()/sum(hist)

    return hist_darkness

def ImageContrast(image):
    '''Uses an entropy filter to measure the contrast of an image

    Keyword Arguments:
    image -- The BGR image collected by UrlToImage()

    Output:
    max_entropy  --  The maximum value of entropy found in the image.
                     Higher values correspond to more contrast.
    '''

    entropy_img = entropy(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), disk(7))
    max_entropy = entr_img.max()

    return max_entropy


def ImageLuminance(r_mean,g_mean,b_mean):
    '''Calculates the standard and percieved luminace of an image.

    Keyword Arguments:
    r_mean -- The average value of the red channel.
    g_mean -- The average value of the green channel.
    b_mean -- The average value of the blue channel.

    Output:
    standard_luminance -- The standard luminance of the image
    percieved_luminance -- The percieved luminace of the image

    Source: 
    http://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
    '''
    standard_luminance  = (0.2126*r_mean + 0.7152*g_mean + 0.0722*b_mean)
    percieved_luminance = (0.299*r_mean + 0.587*g_mean + 0.114*b_mean)

    return standard_luminance, percieved_luminance

def ImageSaliency(saliencymap,h,s,v):
    '''Uses a saliency map to determine HSV values at the focal point
    of the image.  The focal point is defined to be areas above
    a certain saliency level.

    Keyword Arguments:
    saliencymap -- The saliency map of the image.
    h -- The hue channel of the image, in degrees
    s -- The saturation channel of the image
    v -- The value channel of the image

    Output:
    salient_h -- The average hue at the focal points, in degrees
    salient_s -- The average saturation at the focal points
    salient_v -- The average value at the focal points
    '''

    saliency_thresh = 0.2
    salient_h = np.log(circstat.mean(h[saliencymap>=saliency_thresh]*np.pi/180.0)/circstat.mean(h*np.pi/180.0))
    salient_s = np.log(np.mean(s[saliencymap>=saliency_thresh])/np.mean(s))
    salient_v = np.log(np.mean(v[saliencymap>=saliency_thresh])/np.mean(v))

    return salient_h, salient_s, salient_v

def RuleOfThirdsMask(single_channel):
    '''Finds the horizontal and vertical lines of thirds of the image.
    Builds a mask that extract image pixels that are near the rule of
    thirds intersections. 

    Keyword Arguments:
    single_channel -- Any NxM single channel of the image. (Used to image dimensions only)

    Output:
    thrds_mask       -- The NxM mask filled with 1s at the intersection of thirds, and zero elsewhere.
    first_thrd_rows  -- The index of the first line of thirds in the vertical direction.
    second_thrd_rows -- The index of the second line of thirds in the vertical direction.
    first_thrd_cols  -- The index of the first line of thirds in the horizontal direction.
    second_thrd_cols -- The index of the second line of thirds in the horizontal direction.
    '''

    #Get number of rows and columns
    nrows = single_channel.shape[0]
    ncols = single_channel.shape[1]

    #Get 1/3rd and 2/3rd row and columns
    first_thrd_rows  = np.int(np.floor(nrows*1.0/3.0))
    second_thrd_rows = np.int(np.floor(nrows*2.0/3.0))
    first_thrd_cols  = np.int(np.floor(ncols*1.0/3.0))
    second_thrd_cols = np.int(np.floor(ncols*2.0/3.0))

    #Define areas that are "close" to 1/3rd lines
    margin = 20.0
    above_first_thrd_rows  = np.int(first_thrd_rows - np.floor(nrows/margin))
    below_first_thrd_rows  = np.int(first_thrd_rows + np.floor(nrows/margin))

    above_second_thrd_rows = np.int(second_thrd_rows - np.floor(nrows/margin)) #_i
    below_second_thrd_rows = np.int(second_thrd_rows + np.floor(nrows/margin)) #_o

    left_first_thrd_cols   = np.int(first_thrd_cols - np.floor(ncols/margin))
    right_first_thrd_cols  = np.int(first_thrd_cols + np.floor(ncols/margin))

    left_second_thrd_cols  = np.int(second_thrd_cols - np.floor(ncols/margin))
    right_second_thrd_cols = np.int(second_thrd_cols + np.floor(ncols/margin))

    #Build mask of where center of thirds are
    thrds_mask = np.zeros_like(single_channel)
    thrds_mask[above_first_thrd_rows:below_second_thrd_rows,left_first_thrd_cols:right_second_thrd_cols] = 1
    thrds_mask[below_first_thrd_rows:above_second_thrd_rows,right_first_thrd_cols:left_second_thrd_cols] = 0

    return thrds_mask, first_thrd_rows, second_thrd_rows, first_thrd_cols, second_thrd_cols

def RuleOfThirdsStats(saliencymap,h,s,v,mask_details):
    '''Calculates the average H,S,V, and saliency 
    within the specified rule of thirds mask.

    Keyword Arguments:
    saliencymap -- The saliency map of the image.
    h -- The hue channel of the image, in degrees
    s -- The saturation channel of the image
    v -- The value channel of the image
    mask_details -- The tuple output of RuleOfThirdsMask

    Output:
    thirds_h -- The average hue within the rule of thirds mask
    thirds_s -- The average saturation within the rule of thirds mask
    thirds_v -- The average value within the rule of thirds mask
    thirds_saliency -- The average saliency within the rule of thirds mask
    '''

    thrds_mask, first_thrd_rows, second_thrd_rows, first_thrd_cols, second_thrd_cols = mask_details

    #HSV and Saliency of the thirds lines
    thirds_h  = circstat.mean(h[first_thrd_rows:second_thrd_rows,first_thrd_cols:second_thrd_cols]*np.pi/180.0)*180.0/np.pi 
    thirds_s  = np.mean(s[first_thrd_rows:second_thrd_rows,first_thrd_cols:second_thrd_cols]/255.0)                         
    thirds_v  = np.mean(v[first_thrd_rows:second_thrd_rows,first_thrd_cols:second_thrd_cols]/255.0)                         
    thirds_saliency  = np.sum(saliencymap[thrds_mask==1])/np.sum(thrds_mask)

    return thirds_h, thirds_s, thirds_v, thirds_saliency

def RuleOfThirdsDistance(saliencymap, mask_details):
    '''Calculates how far the highest focal point (from saliency map)
    if from on the the rule of thirds intersections.

    Keyword Arguments:
    saliencymap -- The saliency map of the image.
    mask_details -- The tuple output of RuleOfThirdsMask().

    Output:
    thirds_distance -- The minimum distance between the focal point and the thirds intersections.
    '''

    nrows = saliencymap.shape[0]
    ncols = saliencymap.shape[1]
    thrds_mask, first_thrd_rows, second_thrd_rows, first_thrd_cols, second_thrd_cols = mask_details

    (maxs_y,maxs_x) = np.where(saliencymap == np.max(saliencymap))
    t_rows          = [first_thrd_rows,second_thrd_rows]
    t_cols          = [first_thrd_cols,second_thrd_cols]
    thrds_coords    = list(itertools.product(t_rows, t_cols))

    thirds_distance = np.min([np.sqrt(((maxs_x[0] - thrds[1])/np.float(ncols))**2 + \
                        ((maxs_y[0] - thrds[0])/np.float(nrows))**2) for thrds in thrds_coords]) / np.sqrt(2)

    return thirds_distance

def CalcSymmetry(src,mask_details=None):
    '''Calculates the vertical and horizontal symmetry of an image.

    Keyword Arguments:
    src -- A single NxM channel of the image.
    mask_details -- The tuple output of RuleOfThirdsMask(). If none, calculate the symmemtry of the full image.
                    If specified, the symmetry is calculated with only pixels within the mask.

    Output:
    sym_horizontal_value -- The horizontal symmetry of the image channel.
                            Ranges from 0 to 1, with 1 being perfectly symmetric.
    sym_vertical_value   -- The vertical symmetry of the image channel.
                            Ranges from 0 to 1, with 1 being perfectly symmetric.

    Source:
    https://pdfs.semanticscholar.org/42b5/33621f2defe97cb6894eefbb6126ce1f2691.pdf
    '''

    #Flips the image
    a  = src.astype("float")
    upsidedown = src[::-1,:].astype("float") #Flip upsidedown
    sideways_mirror = src[:,::-1].astype("float") #Flip left/right

    #Calculate symmetry by multplying mirrored images
    fs = (a + sideways_mirror)/2
    fa = (a - sideways_mirror)/2
    if type(mask_details) == tuple:
        mask, first_thrd_rows, second_thrd_rows, first_thrd_cols, second_thrd_cols = mask_details
        sym_horizontal_value = (fs[mask==1]**2).sum()/((fs[mask==1]**2).sum() + (fa[mask==1]**2).sum())
    else:
        sym_horizontal_value = (fs**2).sum()/((fs**2).sum() + (fa**2).sum())

    fs = (a + upsidedown)/2
    fa = (a - upsidedown)/2
    if type(mask_details) == tuple:
        sym_vertical_value = (fs[mask==1]**2).sum()/( (fs[mask==1]**2).sum() + (fa[mask==1]**2).sum())
    else:
        sym_vertical_value = (fs**2).sum()/( (fs**2).sum() + (fa**2).sum())

    #Instead of 0.5-1, scale from 0-1
    sym_horizontal_value = sym_horizontal_value*2 - 1
    sym_vertical_value   = sym_vertical_value*2 - 1

    return sym_horizontal_value, sym_vertical_value

def CalcBusyness(gray):
    ''''Quantifies how busy an image is by using
    the variance of the X and Y coordinates of all contours in the image.

    Keyword Arguments:
    gray -- A grayscale version of the image.
    
    Output:
    busyness -- The variance of the X and Y contour centers, normalized to the
                average X and Y of the contour centers, and added in quadrature.
    number_of_contours   -- The number of contours found in the thresholded image.
    '''

    #Make a thresholded image and find contours within it
    ret3,thresh = cv2.threshold(cv2.GaussianBlur(gray,(5,5),30),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh      = cv2.bitwise_not(thresh)
    cnts        = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts        = cnts[0] if imutils.is_cv2() else cnts[1]

    cXs = np.array([])
    cYs = np.array([])
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if  M["m00"] != 0:
            cXs = np.append(cXs,int(M["m10"] / M["m00"]))
            cYs = np.append(cYs,int(M["m01"] / M["m00"]))

    busyness = ( (cXs.std()/cXs.mean())**2 + (cYs.std()/cYs.mean())**2)**(1/2)  
    number_of_contours = len(cnts)

    return busyness, number_of_contours


def UrlToImage(url):
    ''''Creates a CV2 BGR image from a url.

    Keyword Arguments:
    url -- The web address for the image.
    
    Output:
    image -- The BGR image.
    '''

    resp  = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def ExtractFeatures(image):
    ''''Extracts all features from an image.

    Keyword Arguments:
    image -- The BGR image collected by UrlToImage()
    
    Output:
    features -- A dictionary containing the numerical values of each feature.
    '''

    features = {}

    #Image shape features (before resizing)
    features['Aspect_Ratio'] = float(image.shape[0])/image.shape[1]
    features['Image_Size']   = image.size/3 #Divide by three for RGB

    #Resize image to 800x800
    image = ResizeImage(image)

    #Extract RGB color space
    r,g,b = ExtractRGB(image)

    #Extract grayscale color space
    gray  = ExtractGrayscale(image)

    #Extract HSV color space
    h,s,v,h_rad = ExtractHSV(image)
    
    #Extract LAB color space
    L,A,B = ExtractLAB(image) #Captialized to distinguish from "b" blue channel

    #Extract Saliency map for focal point measurements
    sm = SMap.pySaliencyMap(image.shape[1], image.shape[0])
    saliencymap  = sm.SMGetSM(image)

    #Fraction of white pixels (to identify edited images)
    features['frac_white'] = MeasureWhite(image)

    #Sharpness features
    features['Laplacian_Sharpness'] = LaplacianSharpness(gray)
    features['FFT_Sharpness']       = FFTSharpness(gray)

    #LAB Features
    features['Colorfulness'] = Colorfulness(A,B)
    
    #RGB Features
    features['R_Mean'],features['G_Mean'],features['B_Mean'], \
    features['R_Width'],features['G_Width'],features['B_Width'] = RGBStats(image)

    features['R_xgrad'],features['g_xgrad'],features['b_xgrad'],\
    features['r_ygrad'],features['g_ygrad'],features['b_ygrad'],\
    features['r_xgrad_std'],features['g_xgrad_std'],features['b_xgrad_std'],\
    features['r_ygrad_std'],features['g_ygrad_std'],features['b_ygrad_std'] = RGBGradients(r,g,b)

    #HSV Features
    features['H_mean'],features['S_mean'],features['V_mean'],\
    features['H_var'],features['S_var'],features['V_var']  = HSVStats(h_rad,s,v)

    features['Lapacian_Hue'],features['Lapacian_Saturation'],features['Lapacian_Value'] \
        = HSVBlur(h,s,v)

    features['Complimentary_Color_Level'] = ComplimentaryColors(h_rad) 

    #Darkness features
    features['Histogram_Darkness'] = HistogramDarkness(image)
    features['standard_luminance'],features['percieved_luminace'] \
        = ImageLuminance(features['R_Mean'],features['G_Mean'],features['B_Mean'])

    #Focal point features
    features['Salient_Hue'],features['Salient_Saturation'],features['Salient_Value'] \
        = ImageSaliency(saliencymap,h,s,v)

    #Rule of Thirds features
    mask_details = RuleOfThirdsMask(h)

    features['Thirds_Hue'],features['Thirds_Sat'],features['Thirds_Value'],\
        features['Thirds_Saliency'] = RuleOfThirdsStats(saliencymap,h,s,v,mask_details)

    features['Thirds_To_Focal_Distance'] = RuleOfThirdsDistance(saliencymap, mask_details)

    #Symmetry Features
    features['Horizontal_Hue_Sym'],features['Vertical_Hue_Sym']               = CalcSymmetry(h)
    features['Horizontal_Saturation_Sym'],features['Vertical_Saturation_Sym'] = CalcSymmetry(s)
    features['Horizontal_Value_Sym'],features['Vertical_Value_Sym']           = CalcSymmetry(v)

    features['Thirds_Horizontal_Hue_Sym'],features['Thirds_Vertical_Hue_Sym']               = CalcSymmetry(h,mask_details)
    features['Thirds_Horizontal_Saturation_Sym'],features['Thirds_Vertical_Saturation_Sym'] = CalcSymmetry(s,mask_details)
    features['Thirds_Horizontal_Value_Sym'],features['Thirds_Vertical_Value_Sym']           = CalcSymmetry(v,mask_details)
    features['Thirds_Horizontal_Saliency_Sym'],features['Thirds_Vertical_Saliency_Sym']     = CalcSymmetry(saliencymap,mask_details)

    #Image Busyness Features
    features['Busyness'],features['Number_of_Contours'] = CalcBusyness(gray)
    
    return features

def ClassifyImage(image):
    ''''Calculates the probability that an image is "high quality."

    Keyword Arguments:
    image -- The BGR image collected by UrlToImage()
    
    Output:
    high_qual_class -- The predicted class of the image. "True" if the image is classified as high quality.
    high_qual_prob -- The probability of an image belonging to the high quality class.
    '''

    with open('image_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('image_classifier_features.pkl', 'rb') as f:
        image_classifier_features = pickle.load(f)

    #Extract features and use classifier to predict probability of high quality
    feats = ExtractFeatures(image)
    feats = pd.DataFrame(feats,index=[0])
    high_qual_class = clf.predict(feats[image_classifier_features])[0]
    high_qual_prob  = clf.predict_proba(feats[image_classifier_features])[0][1]

    return high_qual_class, high_qual_prob
