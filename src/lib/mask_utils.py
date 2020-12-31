from imutils import face_utils
from PIL import Image
import imutils
import numpy as np
import collections
import dlib
import cv2
import argparse

class MaskUtils:

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('data/dlib/sp68fl.dat')

    def mask_face(self, image, is_file = True):
        if is_file: image = cv2.imread(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out_face = np.zeros_like(image)
        rects = self.detector(gray, 1)

        if len(rects) == 0: return False, False

        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            remapped_shape = np.zeros_like(shape) 
            feature_mask = np.zeros((image.shape[0], image.shape[1])) 

            remapped_shape = self.face_remap(shape)
            cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
            feature_mask = feature_mask.astype(np.bool)
            out_face[feature_mask] = image[feature_mask]

        gray = cv2.cvtColor(out_face, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 1, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [biggest_contour], -1, (255, 255, 255), cv2.FILLED)
        
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        return (image, mask)

    def face_remap(self, shape):
        remapped_image = cv2.convexHull(shape)
        return remapped_image