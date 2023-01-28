from utils import *
import cv2
from fingerprint_enhancer import enhance_Fingerprint

def prepare_dataset(file_names):
    '''
    Coversion to grayscale and enhancement. Split into training and test set.
    :param file_names: All fingerprint images as file names
    :return: train_set, test_set: 2 dictionaries for training and test,
             where the key is the name of the image and the value is the image itself
    '''
    train_set = {}
    test_set = {}
    data = []  # list of tuples
    temp_label = get_image_class(file_names[0])  # sets the image class (101)
    # print(file_names.__len__())

    def edgedet(img):
        k= np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        convolve = cv2.filter2D(gray, -1, k)
        return convolve

    for filename in file_names:
        # print('1')
        # print(filename)
        filename = r'/Users/bhavi/Desktop/preprop/Database/Final_JPEG/Camera/' + filename
        img = cv2.imread(filename)
        # print(img)
        image_gray = grayscale_image(img)
        img = edgedet(img)
        image_gray = (image_gray-np.min(image_gray[:]))/np.ptp(image_gray[:])
        clahe_gs = 29; cl_clip_lim = 5;
        clahe = cv2.createCLAHE(clipLimit=cl_clip_lim, tileGridSize=(clahe_gs,clahe_gs))
        image_clahe = clahe.apply((255*image_gray).astype('uint8')).astype('float'); 
        print('--------')
        image_clahe = (image_clahe-np.min(image_clahe))/np.ptp(image_clahe[:]);
        ATG_image = cv2.adaptiveThreshold(((255*image_clahe).astype('uint8')),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,1)
        gray_img = grayscale_image(ATG_image)
        img = enhance_Fingerprint(gray_img)
        # img = enhance_image(gray_img)
        label = get_image_label(filename)
        print('Processing image {} ...  '.format(label))
        if temp_label != get_image_class(filename):
            train, test = split_dataset(data, 0.2)
            train_set.update(train)
            test_set.update(test)
            temp_label = get_image_class(filename)
            data = []
        data.append((label, img))

        if filename == file_names[len(file_names) - 1]:
            train, test = split_dataset(data, 0.2)
            train_set.update(train)
            test_set.update(test)

    print('DONE')
    return train_set, test_set


def prepare_dataset_authentication(train_feature_descriptors):
    '''
    Splits dataset  by each person data
    for the authentication scenario
    :param train_feature_descriptors: training set
    :return: dictionary where the key denotes the name of the person,
    and the value is a list with all trained feature descriptors
    '''
    authentication_databases = {}
    temp_list = {}
    class_name = get_image_class(list(train_feature_descriptors.keys())[0])  # inital class name
    last_key = list(train_feature_descriptors.keys())[-1]

    for image_id, feature_descriptor in train_feature_descriptors.items():
        if class_name != get_image_class(image_id):
            # print(class_name)
            authentication_databases[class_name] = temp_list
            temp_list = {}
        temp_list[image_id] = feature_descriptor
        class_name = get_image_class(image_id)

        if last_key == image_id:
            authentication_databases[class_name] = temp_list

    return authentication_databases
