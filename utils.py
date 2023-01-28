import cv2
import glob
import numpy
import os
from fingerprintEnhancement import * 
from sklearn.model_selection import train_test_split
from skimage.morphology import skeletonize, thin


def removedot(invertThin):
    temp0 = numpy.array(invertThin[:])
    temp0 = numpy.array(temp0)
    temp1 = temp0/255
    temp2 = numpy.array(temp1)
    temp3 = numpy.array(temp2)
    
    enhanced_img = numpy.array(temp0)
    filter0 = numpy.zeros((10,10))
    W,H = temp0.shape[:2]
    filtersize = 6
    
    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize,j:j + filtersize]

            flag = 0
            if sum(filter0[:,0]) == 0:
                flag +=1
            if sum(filter0[:,filtersize - 1]) == 0:
                flag +=1
            if sum(filter0[0,:]) == 0:
                flag +=1
            if sum(filter0[filtersize - 1,:]) == 0:
                flag +=1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = numpy.zeros((filtersize, filtersize))

    return temp2


def read_images():
    '''
    Reads all images from IMAGES_PATH and sorts them
    :return: sorted file names
    '''
    #file_names = os.listdir(r'C:\Users\VINAY IITB\Downloads/db_2')
    # file_names = os.listdir(r'/Users/bhavi/Desktop/preprop/dbc')
    # file_names = os.listdir(r'/Users/bhavi/Desktop/preprop/Database/Final_JPEG/Scanner/')
    file_names = os.listdir(r'/Users/bhavi/Desktop/preprop/Database/Final_JPEG/Camera/')
    file_names.sort()
    file_names = file_names[:40]
    return file_names

def get_image_label(filename):
    image = filename.split('/')
    # print(image[len(image)-1])
    return image[len(image)-1]


def get_image_class(filename):
    return get_image_label(filename).split('_')[0]


# Splits the dataset on training and testing set
def split_dataset(data, test_size):
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    return train, test


def grayscale_image(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# Enhancement using orientation/frequency filtering - Gabor filterbank
def enhance_image(image):
    image_enhancer = FingerprintImageEnhancer()
    img_e= image_enhancer.enhance(image)
    img_e = numpy.array(img_e, dtype=numpy.uint8)
    ret, img_e = cv2.threshold(img_e, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img_e[img_e == 255] = 1
    skeleton = skeletonize(img_e)
    skeleton = numpy.array(skeleton, dtype=numpy.uint8)
    skeleton = removedot(skeleton)
    harris_corners = cv2.cornerHarris(img_e, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=0)
    return harris_normalized


def get_genuine_impostor_scores(all_scores, identical):
    '''
    Returns two arrays with the genuine and impostor scores.
    The genuine match scores are obtained by matching feature sets
    of the same class (same person) and the impostor match scores are obtained
    by matching feature sets of different classes (different persons)
    '''
    genuine_scores = []
    impostor_scores = []
    for i in range(0, len(all_scores)):
        if identical[i] == 1:
            genuine_scores.append(all_scores[i][1])
        else:
            impostor_scores.append(all_scores[i][1])

    return genuine_scores, impostor_scores




