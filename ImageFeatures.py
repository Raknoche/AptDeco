import argparse
import cv2
from matplotlib import pyplot as plt
import imutils
import numpy as np
import pandas as pd
import pycircstat.descriptive as circstat
import pySaliencyMap as SMap
import itertools
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
import urllib


def CalcSymmetry(src,mask=None):
    a = src.astype("float")
    b1 = src[::-1,:].astype("float") #Flip upsidedown
    b2 = src[:,::-1].astype("float") #Flip left/right

    #Calculate symmetry by multplying mirrored images
    fs = (a + b2)/2
    fa = (a - b2)/2
    if type(mask) == np.ndarray:
        Sym_Horizontal_Value = (fs[mask==1]**2).sum()/((fs[mask==1]**2).sum() + (fa[mask==1]**2).sum())
    else:
        Sym_Horizontal_Value = (fs**2).sum()/((fs**2).sum() + (fa**2).sum())

    fs = (a + b1)/2
    fa = (a - b1)/2
    if type(mask) == np.ndarray:
        Sym_Vertical_Value = (fs[mask==1]**2).sum()/( (fs[mask==1]**2).sum() + (fa[mask==1]**2).sum())
    else:
        Sym_Vertical_Value = (fs**2).sum()/( (fs**2).sum() + (fa**2).sum())

    #Instead of 0.5-1, scale from 0-1
    Sym_Horizontal_Value=Sym_Horizontal_Value*2 - 1
    Sym_Vertical_Value=Sym_Vertical_Value*2 - 1

    return(Sym_Horizontal_Value,Sym_Vertical_Value)

def ExtractFeatures(image):
    '''Input is cv2 image'''

    features={}

    '''Image Shape Features -- SHOULD ALL BE 200x200''' 
    #features['Aspect_Ratio'] = float(image.shape[0])/image.shape[1]
    #features['Image_Size'] = image.size/3 #Divide by three for RGB


    #Resize image (should already be 200x200)
    scaler = np.min([800.0/image.shape[0], 800.0/image.shape[1]])
    image = cv2.resize(image,(np.int(scaler*image.shape[1]),np.int(scaler*image.shape[0])),interpolation=cv2.INTER_AREA)

    #Extract color spaces
    b=image[:,:,0]
    g=image[:,:,1]
    r=image[:,:,2]

    I_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    I_hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    I_h = I_hsv[:,:,0]
    I_s = I_hsv[:,:,1]
    I_v = I_hsv[:,:,2]
    I_h_rad = I_h.flatten()*np.pi/180.0 #Hue converted to radians

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel,a_channel,b_channel = cv2.split(lab_image)

    #Fraction of white pixels (to identify edited images)
    white_pixels = (image == 255).all(axis=2)
    features['frac_white'] = len(white_pixels[white_pixels==True])/white_pixels.size


    '''Sharpness features'''
    #Feature 1
    laplacian = cv2.Laplacian(I_gray, cv2.CV_64F)
    features['Laplacian_Sharpness']= laplacian.var()

    #Feature 2
    rows, cols = I_gray.shape
    crow, ccol = int(rows/2),int(cols/2)
    f = np.fft.fft2(I_gray)
    fshift = np.fft.fftshift(f)
    fshift[crow-75:crow+75, ccol-75:ccol+75] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_fft = np.fft.ifft2(f_ishift)
    img_fft = 20*np.log(np.abs(img_fft))
    features['FFT_Sharpness'] = np.mean(img_fft)


    '''Color Features'''
    #Feature 4 - Is it gray
    #features['IsGray'] = np.all(I_s==0)

    #Feature X -- How colorful is it
    mu_ab = np.sqrt( a_channel.mean()**2 + b_channel.mean()**2)
    features['Colorfulness'] = a_channel.std() + b_channel.std() + 0.39*mu_ab


    #Feature 5-10 - Avg and normalized standard deviation of each color channel
    (rgb_means, rgb_stds) = cv2.meanStdDev(image)
    features['B_Mean'],features['G_Mean'],features['R_Mean']=rgb_means[:,0]
    features['B_Width'],features['G_Width'],features['R_Width']=rgb_stds[:,0]/rgb_means[:,0]

    #Feature 11-22 - Mean and standard deviation of color gradients in each channel
    features['R_xgrad'] = np.mean(cv2.Sobel(r,cv2.CV_64F,1,0,ksize=1))
    features['g_xgrad']= np.mean(cv2.Sobel(g,cv2.CV_64F,1,0,ksize=1))
    features['b_xgrad'] = np.mean(cv2.Sobel(b,cv2.CV_64F,1,0,ksize=1))

    features['r_ygrad'] = np.mean(cv2.Sobel(r,cv2.CV_64F,0,1,ksize=1))
    features['g_ygrad'] = np.mean(cv2.Sobel(g,cv2.CV_64F,0,1,ksize=1))
    features['b_ygrad'] = np.mean(cv2.Sobel(b,cv2.CV_64F,0,1,ksize=1))

    features['r_xgrad_std'] = np.std(cv2.Sobel(r,cv2.CV_64F,1,0,ksize=1))
    features['g_xgrad_std'] = np.std(cv2.Sobel(g,cv2.CV_64F,1,0,ksize=1))
    features['b_xgrad_std'] = np.std(cv2.Sobel(b,cv2.CV_64F,1,0,ksize=1))

    features['r_ygrad_std'] = np.std(cv2.Sobel(r,cv2.CV_64F,0,1,ksize=1))
    features['g_ygrad_std'] = np.std(cv2.Sobel(g,cv2.CV_64F,0,1,ksize=1))
    features['b_ygrad_std'] = np.std(cv2.Sobel(b,cv2.CV_64F,0,1,ksize=1))

    #Feautres XX - HSV characteristic
    features['H_mean'] = circstat.mean(I_h_rad)*180.0/np.pi           
    features['H_var']  = circstat.var(I_h_rad)*180.0/np.pi

    features['S_mean'] = np.mean(I_s)/255.0                           
    features['S_var']  = np.var(I_s/255.0)

    features['V_mean'] = np.mean(I_v)/255.0                           
    features['V_var']  = np.var(I_v/255.0)

    features['Lapacian_Hue'] = cv2.Laplacian(I_h/255.0, cv2.CV_64F).var()
    features['Lapacian_Saturation'] = cv2.Laplacian(I_s/255.0, cv2.CV_64F).var()
    features['Lapacian_Value']     = cv2.Laplacian(I_v/255.0, cv2.CV_64F).var()

    #Feature XX -- complementary colors
    features['Complimentary_Color_Level'] = np.abs(np.exp(2*I_h_rad*1j).sum() / len(I_h.flatten())) #ranges from 0 to 1, 1 is more complementary


    '''Darkness Features'''
    #Feature 3
    hist,bins = np.histogram(image.ravel(),255,[0,255])
    bin_centers = (bins[1:]+bins[:-1])/2
    features['Histogram_Darkness'] = (bin_centers*hist).sum()/sum(hist)

    #Feature -- Contrast level (takes too long.. goes from 0.3 -> 1.2 seconds)
    #entr_img = entropy(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), disk(7))
    #features['Contrast']=entr_img.max()

    #Feature 
    features['standard_luminance'] = (0.2126*features['R_Mean'] + 0.7152*features['G_Mean'] + 0.0722*features['B_Mean'])
    features['percieved_luminace'] = (0.299*features['R_Mean'] + 0.587*features['G_Mean'] + 0.114*features['B_Mean'])


    '''Focal Point Features'''
    sm = SMap.pySaliencyMap(image.shape[1], image.shape[0])
    saliencymap    = sm.SMGetSM(image)

    #Features XX -- HSV at high saliency focal points, compared to image average
    Saliency_Thresh = 0.2
    features['Salient_Hue'] = np.log(circstat.mean(I_h[saliencymap>=Saliency_Thresh]*np.pi/180.0)/circstat.mean(I_h*np.pi/180.0))
    features['Salient_Saturation'] = np.log(np.mean(I_s[saliencymap>=Saliency_Thresh])/np.mean(I_s))
    features['Salient_Value'] = np.log(np.mean(I_v[saliencymap>=Saliency_Thresh])/np.mean(I_v))


    '''Rule of Thirds Features'''
    #Get number of row and columns
    nrows = image.shape[0]
    ncols = image.shape[1]

    #Get 1/3rd and 2/3rd row and columns
    first_thrd_rows = np.int(np.floor(nrows*1.0/3.0))
    second_thrd_rows = np.int(np.floor(nrows*2.0/3.0))
    first_thrd_cols = np.int(np.floor(ncols*1.0/3.0))
    second_thrd_cols = np.int(np.floor(ncols*2.0/3.0))

    #Define areas that are "close" to 1/3rd lines
    margin = 20.0
    above_first_thrd_rows = np.int(first_thrd_rows - np.floor(nrows/margin))
    below_first_thrd_rows = np.int(first_thrd_rows + np.floor(nrows/margin))

    above_second_thrd_rows = np.int(second_thrd_rows - np.floor(nrows/margin)) #_i
    below_second_thrd_rows = np.int(second_thrd_rows + np.floor(nrows/margin)) #_o

    left_first_thrd_cols = np.int(first_thrd_cols - np.floor(ncols/margin))
    right_first_thrd_cols = np.int(first_thrd_cols + np.floor(ncols/margin))

    left_second_thrd_cols = np.int(second_thrd_cols - np.floor(ncols/margin))
    right_second_thrd_cols = np.int(second_thrd_cols + np.floor(ncols/margin))

    #Build mask of where center of thirds are
    thrds_mask = np.zeros_like(I_h)
    thrds_mask[above_first_thrd_rows:below_second_thrd_rows,left_first_thrd_cols:right_second_thrd_cols] = 1
    thrds_mask[below_first_thrd_rows:above_second_thrd_rows,right_first_thrd_cols:left_second_thrd_cols] = 0

    #HSV and Saliency of the thirds lines
    features['Thirds_Hue']      = circstat.mean(I_h[first_thrd_rows:second_thrd_rows,first_thrd_cols:second_thrd_cols]*np.pi/180.0)*180.0/np.pi 
    features['Thirds_Sat']      = np.mean(I_s[first_thrd_rows:second_thrd_rows,first_thrd_cols:second_thrd_cols]/255.0)                         
    features['Thirds_Value']    = np.mean(I_v[first_thrd_rows:second_thrd_rows,first_thrd_cols:second_thrd_cols]/255.0)                         
    features['Thirds_Saliency'] = np.sum(saliencymap[thrds_mask==1])/np.sum(thrds_mask)

    #How far is the maximum focal point from the thirds intersections
    (maxs_y,maxs_x) = np.where(saliencymap == np.max(saliencymap))
    t_rows = [first_thrd_rows,second_thrd_rows]
    t_cols = [first_thrd_cols,second_thrd_cols]
    thrds_coords = list(itertools.product(t_rows, t_cols))
    features['Thirds_To_Focal_Distance']= np.min([np.sqrt(((maxs_x[0] - thrds[1])/np.float(ncols))**2 + ((maxs_y[0] - thrds[0])/np.float(nrows))**2) for thrds in thrds_coords]) / np.sqrt(2)

    '''Symmetry Features'''
    features['Horizontal_Hue_Sym'],features['Vertical_Hue_Sym'] = CalcSymmetry(I_h)
    features['Horizontal_Saturation_Sym'],features['Vertical_Saturation_Sym'] = CalcSymmetry(I_s)
    features['Horizontal_Value_Sym'],features['Vertical_Value_Sym'] = CalcSymmetry(I_v)

    features['Thirds_Horizontal_Hue_Sym'],features['Thirds_Vertical_Hue_Sym'] = CalcSymmetry(I_h,thrds_mask)
    features['Thirds_Horizontal_Saturation_Sym'],features['Thirds_Vertical_Saturation_Sym'] = CalcSymmetry(I_s,thrds_mask)
    features['Thirds_Horizontal_Value_Sym'],features['Thirds_Vertical_Value_Sym'] = CalcSymmetry(I_v,thrds_mask)
    features['Thirds_Horizontal_Saliency_Sym'],features['Thirds_Vertical_Saliency_Sym'] = CalcSymmetry(saliencymap,thrds_mask)

    ''' Image Busyness '''
    ret3,thresh = cv2.threshold(cv2.GaussianBlur(I_gray,(5,5),30),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh=cv2.bitwise_not(thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    cXs=np.array([])
    cYs=np.array([])
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if  M["m00"] != 0:
            cXs = np.append(cXs,int(M["m10"] / M["m00"]))
            cYs = np.append(cYs,int(M["m01"] / M["m00"]))

    features['Busyness'] = ( (cXs.std()/cXs.mean())**2 + (cYs.std()/cYs.mean())**2)**(1/2)  
    features['Number_of_Contours'] = len(cnts)
    
    return features


def ExtractFeatures_NoBackground(image):

    features={}


    '''Image Shape Features'''
    #features['Aspect_Ratio'] = float(image.shape[0])/image.shape[1]
    #features['Image_Size'] = image.size/3 #Divide by three for RGB


    #Resize image from 200x200
    scaler = np.min([800.0/image.shape[0], 800.0/image.shape[1]])
    image = cv2.resize(image,(np.int(scaler*image.shape[1]),np.int(scaler*image.shape[0])),interpolation=cv2.INTER_AREA)
    I_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    #Find contour
    ret3,thresh = cv2.threshold(cv2.GaussianBlur(I_gray,(5,5),30),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh=cv2.bitwise_not(thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    cXs=np.array([])
    cYs=np.array([])
    
    #mask to select within contour bounds
    mask = thresh.astype(bool)
    
    #Arrays for contour features
    areas = np.array([]) #fraction of image occupied by object
    widths = np.array([]) #item widths (normalized to image width)
    heights = np.array([]) #item heights (normalized to image height)
    extents = np.array([]) #fraction of rectangular bounding box that is occupied
    left_weights = np.array([]) #(x_center - x_min)/width -- how left heavy is it
    top_weights = np.array([]) #(y_center - y_min)/height -- how top heavy is it
    aspect_ratios = np.array([]) #ratio of width to height
    solidities = np.array([]) #How "solid" the object is
    
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if  M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            #contour properties
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            rect_area = w*h
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)

            #Furniture Features
            areas = np.append(areas,float(area))
            extents = np.append(extents,float(area)/rect_area)
            widths = np.append(widths,float(w)/image.shape[1])
            heights =np.append(heights,float(h)/image.shape[0])
            left_weights = np.append(left_weights, (cX - x)/w)
            top_weights = np.append(top_weights, (cY - y)/w)
            aspect_ratios = np.append(aspect_ratios, float(w)/h)
            solidities = np.append(solidities,float(area)/hull_area)
            cXs = np.append(cXs,int(M["m10"] / M["m00"]))
            cYs = np.append(cYs,int(M["m01"] / M["m00"]))

    #Full area of the contours
    features['Fractional_Contour_Area'] = areas.sum()/thresh.size
   
    #Image dimensions
    half_xl = np.floor(thresh.shape[1]/2)
    half_yl = np.floor(thresh.shape[0]/2)

    features['Centroid_XOffset'] = (cX-half_xl)/(half_xl*2)
    features['Centroid_YOffset'] = (cY-half_yl)/(half_yl*2)
    features['Centroid_to_Center_Distance'] = (((cY-half_yl)/(half_yl*2))**2 + ((cX-half_xl)/(half_xl*2))**2)**(1/2)
    features['Fractional_BB_Contour_Area'] = extents.mean()
    features['Contour_Solidity'] = solidities.mean()
    features['Contour_Width'] = widths.mean()
    features['Contour_Height'] = heights.mean()
     
            
    
    #Extract color spaces
    b=image[:,:,0]
    g=image[:,:,1]
    r=image[:,:,2]

    #Fraction of white pixels (to identify edited images)
    white_pixels = (image == 255).all(axis=2)
    features['frac_white'] = len(white_pixels[white_pixels==True])/white_pixels.size


    I_hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    I_h = I_hsv[:,:,0]
    I_s = I_hsv[:,:,1]
    I_v = I_hsv[:,:,2]
    I_h_rad = I_h*np.pi/180.0 #Hue converted to radians

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel,a_channel,b_channel = cv2.split(lab_image)


    '''Sharpness features'''
    #Feature 1
    laplacian = cv2.Laplacian(I_gray, cv2.CV_64F)
    features['Laplacian_Sharpness']= laplacian[mask==1].var()

    #Feature 2
    rows, cols = I_gray.shape
    crow, ccol = int(rows/2),int(cols/2)
    f = np.fft.fft2(I_gray)
    fshift = np.fft.fftshift(f)
    fshift[crow-75:crow+75, ccol-75:ccol+75] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_fft = np.fft.ifft2(f_ishift)
    img_fft = 20*np.log(np.abs(img_fft))
    features['FFT_Sharpness'] = np.mean(img_fft[mask==1])


    '''Color Features'''
    #Feature 4 - Is it gray
    #features['IsGray'] = np.all(I_s[mask==1]==0)

    #Feature X -- How colorful is it
    mu_ab = np.sqrt( a_channel[mask==1].mean()**2 + b_channel[mask==1].mean()**2)
    features['Colorfulness'] = a_channel[mask==1].std() + b_channel[mask==1].std() + 0.39*mu_ab


    #Feature 5-10 - Avg and normalized standard deviation of each color channel
    features['B_Mean'] = b[mask==1].mean()
    features['G_Mean'] = g[mask==1].mean()
    features['R_Mean'] = r[mask==1].mean()
    features['B_Width'] = b[mask==1].std()/features['B_Mean']
    features['G_Width'] = g[mask==1].std()/features['G_Mean']
    features['R_Width'] = r[mask==1].std()/features['R_Mean']

    #Feature 11-22 - Mean and standard deviation of color gradients in each channel
    features['R_xgrad'] = np.mean(cv2.Sobel(r,cv2.CV_64F,1,0,ksize=1)[mask==1])
    features['g_xgrad']= np.mean(cv2.Sobel(g,cv2.CV_64F,1,0,ksize=1)[mask==1])
    features['b_xgrad'] = np.mean(cv2.Sobel(b,cv2.CV_64F,1,0,ksize=1)[mask==1])

    features['r_ygrad'] = np.mean(cv2.Sobel(r,cv2.CV_64F,0,1,ksize=1)[mask==1])
    features['g_ygrad'] = np.mean(cv2.Sobel(g,cv2.CV_64F,0,1,ksize=1)[mask==1])
    features['b_ygrad'] = np.mean(cv2.Sobel(b,cv2.CV_64F,0,1,ksize=1)[mask==1])

    features['r_xgrad_std'] = np.std(cv2.Sobel(r,cv2.CV_64F,1,0,ksize=1)[mask==1])
    features['g_xgrad_std'] = np.std(cv2.Sobel(g,cv2.CV_64F,1,0,ksize=1)[mask==1])
    features['b_xgrad_std'] = np.std(cv2.Sobel(b,cv2.CV_64F,1,0,ksize=1)[mask==1])

    features['r_ygrad_std'] = np.std(cv2.Sobel(r,cv2.CV_64F,0,1,ksize=1)[mask==1])
    features['g_ygrad_std'] = np.std(cv2.Sobel(g,cv2.CV_64F,0,1,ksize=1)[mask==1])
    features['b_ygrad_std'] = np.std(cv2.Sobel(b,cv2.CV_64F,0,1,ksize=1)[mask==1])

    #Feautres XX - HSV characteristic
    features['H_mean'] = circstat.mean(I_h_rad[mask==1])*180.0/np.pi           
    features['H_var']  = circstat.var(I_h_rad[mask==1])*180.0/np.pi

    features['S_mean'] = np.mean(I_s[mask==1])/255.0                           
    features['S_var']  = np.var(I_s[mask==1]/255.0)

    features['V_mean'] = np.mean(I_v[mask==1])/255.0                           
    features['V_var']  = np.var(I_v[mask==1]/255.0)

    features['Lapacian_Hue'] = cv2.Laplacian(I_h/255.0, cv2.CV_64F)[mask==1].var()
    features['Lapacian_Saturation'] = cv2.Laplacian(I_s/255.0, cv2.CV_64F)[mask==1].var()
    features['Lapacian_Value']     = cv2.Laplacian(I_v/255.0, cv2.CV_64F)[mask==1].var()

    #Feature XX -- complementary colors
    features['Complimentary_Color_Level'] = np.abs(np.exp(2*I_h_rad[mask==1]*1j).sum() / len(I_h[mask==1].flatten())) #ranges from 0 to 1, 1 is more complementary


    '''Darkness Features'''
    #Feature 3
    hist,bins = np.histogram(image.ravel(),255,[0,255])
    bin_centers = (bins[1:]+bins[:-1])/2
    features['Histogram_Darkness'] = (bin_centers*hist).sum()/sum(hist)

    #Feature -- Contrast level (takes too long.. goes from 0.3 -> 1.2 seconds)
    #entr_img = entropy(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), disk(7))
    #features['Contrast']=entr_img[mask==1].max()

    #Feature 
    features['standard_luminance'] = (0.2126*features['R_Mean'] + 0.7152*features['G_Mean'] + 0.0722*features['B_Mean'])
    features['percieved_luminace'] = (0.299*features['R_Mean'] + 0.587*features['G_Mean'] + 0.114*features['B_Mean'])


    '''Focal Point Features'''
    sm = SMap.pySaliencyMap(image.shape[1], image.shape[0])
    saliencymap    = sm.SMGetSM(image)

    #Features XX -- HSV at high saliency focal points, compared to image average
    Saliency_Thresh = 0.2
    features['Salient_Hue'] = np.log(circstat.mean(I_h[saliencymap>=Saliency_Thresh]*np.pi/180.0)/circstat.mean(I_h*np.pi/180.0))
    features['Salient_Saturation'] = np.log(np.mean(I_s[saliencymap>=Saliency_Thresh])/np.mean(I_s))
    features['Salient_Value'] = np.log(np.mean(I_v[saliencymap>=Saliency_Thresh])/np.mean(I_v))


    '''Rule of Thirds Features'''
    #Get number of row and columns
    nrows = image.shape[0]
    ncols = image.shape[1]

    #Get 1/3rd and 2/3rd row and columns
    first_thrd_rows = np.int(np.floor(nrows*1.0/3.0))
    second_thrd_rows = np.int(np.floor(nrows*2.0/3.0))
    first_thrd_cols = np.int(np.floor(ncols*1.0/3.0))
    second_thrd_cols = np.int(np.floor(ncols*2.0/3.0))

    #Define areas that are "close" to 1/3rd lines
    margin = 20.0
    above_first_thrd_rows = np.int(first_thrd_rows - np.floor(nrows/margin))
    below_first_thrd_rows = np.int(first_thrd_rows + np.floor(nrows/margin))

    above_second_thrd_rows = np.int(second_thrd_rows - np.floor(nrows/margin)) #_i
    below_second_thrd_rows = np.int(second_thrd_rows + np.floor(nrows/margin)) #_o

    left_first_thrd_cols = np.int(first_thrd_cols - np.floor(ncols/margin))
    right_first_thrd_cols = np.int(first_thrd_cols + np.floor(ncols/margin))

    left_second_thrd_cols = np.int(second_thrd_cols - np.floor(ncols/margin))
    right_second_thrd_cols = np.int(second_thrd_cols + np.floor(ncols/margin))

    #Build mask of where center of thirds are
    thrds_mask = np.zeros_like(I_h)
    thrds_mask[above_first_thrd_rows:below_second_thrd_rows,left_first_thrd_cols:right_second_thrd_cols] = 1
    thrds_mask[below_first_thrd_rows:above_second_thrd_rows,right_first_thrd_cols:left_second_thrd_cols] = 0

    #HSV and Saliency of the thirds lines
    features['Thirds_Hue']      = circstat.mean(I_h[first_thrd_rows:second_thrd_rows,first_thrd_cols:second_thrd_cols]*np.pi/180.0)*180.0/np.pi 
    features['Thirds_Sat']      = np.mean(I_s[first_thrd_rows:second_thrd_rows,first_thrd_cols:second_thrd_cols]/255.0)                         
    features['Thirds_Value']    = np.mean(I_v[first_thrd_rows:second_thrd_rows,first_thrd_cols:second_thrd_cols]/255.0)                         
    features['Thirds_Saliency'] = np.sum(saliencymap[thrds_mask==1])/np.sum(thrds_mask)

    #How far is the maximum focal point from the thirds intersections
    (maxs_y,maxs_x) = np.where(saliencymap == np.max(saliencymap))
    t_rows = [first_thrd_rows,second_thrd_rows]
    t_cols = [first_thrd_cols,second_thrd_cols]
    thrds_coords = list(itertools.product(t_rows, t_cols))
    features['Thirds_To_Focal_Distance']= np.min([np.sqrt(((maxs_x[0] - thrds[1])/np.float(ncols))**2 + ((maxs_y[0] - thrds[0])/np.float(nrows))**2) for thrds in thrds_coords]) / np.sqrt(2)

    '''Symmetry Features'''
    features['Horizontal_Hue_Sym'],features['Vertical_Hue_Sym'] = CalcSymmetry(I_h)
    features['Horizontal_Saturation_Sym'],features['Vertical_Saturation_Sym'] = CalcSymmetry(I_s)
    features['Horizontal_Value_Sym'],features['Vertical_Value_Sym'] = CalcSymmetry(I_v)

    features['Thirds_Horizontal_Hue_Sym'],features['Thirds_Vertical_Hue_Sym'] = CalcSymmetry(I_h,thrds_mask)
    features['Thirds_Horizontal_Saturation_Sym'],features['Thirds_Vertical_Saturation_Sym'] = CalcSymmetry(I_s,thrds_mask)
    features['Thirds_Horizontal_Value_Sym'],features['Thirds_Vertical_Value_Sym'] = CalcSymmetry(I_v,thrds_mask)
    features['Thirds_Horizontal_Saliency_Sym'],features['Thirds_Vertical_Saliency_Sym'] = CalcSymmetry(saliencymap,thrds_mask)

    ''' Image Busyness '''
    features['Busyness'] = ( (cXs.std()/cXs.mean())**2 + (cYs.std()/cYs.mean())**2)**(1/2)  
    features['Number_of_Contours'] = len(cnts)
    
    return features

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image