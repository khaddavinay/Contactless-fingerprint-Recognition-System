import cv2
import numpy as np
import skimage.morphology
import matplotlib
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
import math
import glob
from PIL import Image
from PIL import ImageEnhance

class MinutiaeFeature(object):
    def __init__(self, locX, locY, Orientation, Type):
        self.locX = locX;
        self.locY = locY;
        self.Orientation = Orientation;
        self.Type = Type;
    def __repr__(self):
        return f' locX:{self.locX} locY:{self.locY}  Orientation:{self.Orientation}'

class FingerprintFeatureExtractor(object):
    def __init__(self):
        self._mask = []
        self._skel = []
        self.minutiaeTerm = []
        self.minutiaeBif = []

    def __skeletonize(self, img):
        img = np.uint8(img > 128)
        self._skel = skimage.morphology.skeletonize(img)
        self._skel = np.uint8(self._skel) * 255
        self._mask = img * 255

    def __computeAngle(self, block, minutiaeType):
        angle = []
        (blkRows, blkCols) = np.shape(block);
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        if (minutiaeType.lower() == 'termination'):
            sumVal = 0;
            for i in range(blkRows):
                for j in range(blkCols):
                    if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                        sumVal += 1
                        if (sumVal > 1):
                            angle.append(float('nan'))
            return (angle)

        elif (minutiaeType.lower() == 'bifurcation'):
            (blkRows, blkCols) = np.shape(block);
            CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
            angle = []
            sumVal = 0;
            for i in range(blkRows):
                for j in range(blkCols):
                    if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                        sumVal += 1
            if (sumVal != 3):
                angle.append(float('nan'))
            return (angle)

    def __getTerminationBifurcation(self):
        self._skel = self._skel == 255;
        (rows, cols) = self._skel.shape;
        self.minutiaeTerm = np.zeros(self._skel.shape);
        self.minutiaeBif = np.zeros(self._skel.shape);

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if (self._skel[i][j] == 1):
                    block = self._skel[i - 1:i + 2, j - 1:j + 2];
                    block_val = np.sum(block);
                    if (block_val == 2):
                        self.minutiaeTerm[i, j] = 1;
                    elif (block_val == 4):
                        self.minutiaeBif[i, j] = 1;

        self._mask = convex_hull_image(self._mask > 0)
        self._mask = erosion(self._mask, square(5))  # Structuing element for mask erosion = square(5)
        self.minutiaeTerm = np.uint8(self._mask) * self.minutiaeTerm

    def __removeSpuriousMinutiae(self, minutiaeList, img, thresh):#see in sp.text file for more detaile
        img = img * 0;
        SpuriousMin = [];
        numPoints = len(minutiaeList);
        D = np.zeros((numPoints, numPoints))
        for i in range(1,numPoints):
            for j in range(0, i):
                (X1,Y1) = minutiaeList[i]['centroid']
                (X2,Y2) = minutiaeList[j]['centroid']

                dist = np.sqrt((X2-X1)**2 + (Y2-Y1)**2);
                D[i][j] = dist
                if(dist < thresh):
                    SpuriousMin.append(i)
                    SpuriousMin.append(j)

        SpuriousMin = np.unique(SpuriousMin)
        for i in range(0,numPoints):
            if(not i in SpuriousMin):
                (X,Y) = np.int16(minutiaeList[i]['centroid']);
                img[X,Y] = 1;

        img = np.uint8(img);
        return(img)

    def __cleanMinutiae(self, img):
        self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2);#Label connected regions of an integer array
        RP = skimage.measure.regionprops(self.minutiaeTerm)#Measure properties of labeled image regions.
        self.minutiaeTerm = self.__removeSpuriousMinutiae(RP, np.uint8(img), 10);

    def __performFeatureExtraction(self):
        FeaturesTerm = []
        self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2);
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeTerm))

        WindowSize = 2  # --> For Termination, the block size must can be 3x3, or 5x5. Hence the window selected is 1 or 2
        FeaturesTerm = []
        for num, i in enumerate(RP):
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Termination')
            if(len(angle) == 1):
                FeaturesTerm.append(MinutiaeFeature(row, col, angle, 'Termination'))

        FeaturesBif = []
        self.minutiaeBif = skimage.measure.label(self.minutiaeBif, connectivity=2);
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeBif))
        WindowSize = 1  # --> For Bifurcation, the block size must be 3x3. Hence the window selected is 1
        for i in RP:
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Bifurcation')
            if(len(angle) == 3):
                FeaturesBif.append(MinutiaeFeature(row, col, angle, 'Bifurcation'))
        return (FeaturesTerm, FeaturesBif)

    def extractMinutiaeFeatures(self, img):
        self.__skeletonize(img)

        self.__getTerminationBifurcation()

        self.__cleanMinutiae(img)

        FeaturesTerm, FeaturesBif = self.__performFeatureExtraction()
        return(FeaturesTerm, FeaturesBif)

    def showResults(self):
        BifLabel = skimage.measure.label(self.minutiaeBif, connectivity=2);
        TermLabel = skimage.measure.label(self.minutiaeTerm, connectivity=2);

        minutiaeBif = TermLabel * 0;
        minutiaeTerm = BifLabel * 0;

        (rows, cols) = self._skel.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255*self._skel;
        DispImg[:, :, 1] = 255*self._skel;
        DispImg[:, :, 2] = 255*self._skel;

        RP = skimage.measure.regionprops(BifLabel)
        for idx, i in enumerate(RP):
            (row, col) = np.int16(np.round(i['Centroid']))
            minutiaeBif[row, col] = 1;
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3);
            skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0));

        RP = skimage.measure.regionprops(TermLabel)
        for idx, i in enumerate(RP):
            (row, col) = np.int16(np.round(i['Centroid']))
            minutiaeTerm[row, col] = 1;
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3);
            skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255));

        return DispImg


def extract_minutiae_features(img, showResult=False):
    feature_extractor = FingerprintFeatureExtractor()
    FeaturesTerm, FeaturesBif = feature_extractor.extractMinutiaeFeatures(img)

    if(showResult):
        image=feature_extractor.showResults()

    return(FeaturesTerm, FeaturesBif,image)



# def get_minutiae_and_matching(img,showResult=False):
#     FeaturesTerminations, FeaturesBifurcations,DispImg = extract_minutiae_features(img, showResult=True)
#     # font = cv2.FONT_HERSHEY_SIMPLEX
#     # color = (255, 0, 0)
#     # org = (10, 15)
#     # DispImg = cv2.putText(DispImg, 'Bifurcations', org, font,0.5, color,2, cv2.LINE_AA)
#     # color1 = (0, 0, 255)
#     # org1 = (10, 30)
#     # DispImg = cv2.putText(DispImg, 'Terminations', org1, font,0.5, color1,2, cv2.LINE_AA)
#     # plt.subplot(1,3,1), plt.imshow(img,cmap = 'gray')
#     # plt.title('Original Image')
#     # plt.subplot(1,3,2), plt.imshow(DispImg,cmap = 'gray')
#     # plt.title('Minutiae Extraction')
#     #plt.show()
#     #cv2.imshow('Minutiae Points', DispImg)

#     # print()
#     # print("--------------------Terminations Points-----------------------------")
#     # print()
#     # print([FeaturesTerminations])
#     # print()
#     # print("--------------------Bifurcations Points-----------------------------")
#     # print()
#     # print([FeaturesBifurcations])

#     #---------------------------matching with database------------------------------------

#     path="C:/Users/91985/Desktop/6th_sem_project/project/db/*.*"

#     for file in glob.glob(path):

#         img2=cv2.imread(file)

#         # kp1, des1 = edge_processing(img,125)

#         # kp2, des2 = edge_processing(img2,125)

#         # # Match descriptors.
#         # matches = match_edge_descriptors(des2,des1)

#         sift = cv2.xfeatures2d.SIFT_create()
            
#         kp1, descriptors_1 = sift.detectAndCompute(img, None)
#         kp2, descriptors_2 = sift.detectAndCompute(img2, None)

#         matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), 
#                     dict()).knnMatch(descriptors_1, descriptors_2, k=2)
#         # Calculate score
#         # score = sum([match.distance for match in matches])
#         # print(score)
#         match_points = []
        
#         for p, q in matches: #pdf keyPoints_and_matching
#             if p.distance < 0.1*q.distance:
#                 match_points.append(p)
#         keypoints = 0
#         if len(kp1) <= len(kp2):
#             keypoints = len(kp1)            
#         else:
#             keypoints = len(kp2)

#         if (len(match_points) / keypoints)>0.95:
#             print("% match: ", len(match_points) / keypoints * 100)
#             print("Figerprint ID: " + str(file)) 
#             img3 = cv2.drawMatches(img,kp1,img2,kp2,match_points,None)
#             # plt.subplot(1,3,3), plt.imshow(img3)
#             # plt.title('Matched Image')
#             # plt.show()
#             flag=0
#             break
#         else:
#             flag=1
#     # for m in matches: #need to get access to each pair coordinates of matches object 
#     #     p1 = kp1[m.queryIdx].pt
#     #     p2 = kp2[m.trainIdx].pt
#     #     print(p1,p2)
#     if flag==1:
#         print("Not Matched")




				







