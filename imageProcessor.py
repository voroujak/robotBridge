'''
@author: voroujak
'''

import rospy
import cv2
import argparse

from std_msgs.msg import String
#from sensor_msgs.msg import Image as ImageMsg
#from cv_bridge import CvBridge
import numpy as np
from PIL import Image as PILImage
from scipy.misc import toimage
from scipy.misc import imresize
import os
import json
import time
#imports for object recognition
import keras.backend as K
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


class imageProcessor:
    def __init__(self):#, PIP="10.0.0.3", PPORT=9559):
        #K.clear_session()
        self.mode = 0
        #self.PIP = PIP
        #self.PPORT = PPORT
        K.clear_session()
        
        self.modelImage = ResNet50(weights='imagenet')
        #self.modelImage._make_predict_function()
        self.width = 224
        self.height = 224
        self.stackedImages = []
        #predictedLabels =self.predictObjects([1,1])#image)
        #rospy.init_node("imageProcessor")
        self.params = {'quality':2, 'colorSpace':11, 'nuImagesSequence' : 1, 'savePath' : "imagesCache/", 'fps' : 5}
        self.trigPub = rospy.Publisher("/naoqi_camera/captureTrigger", String, queue_size=1)
        time.sleep(0.5) # otherwise, it wont publish message
        #print(str(self.params))
        for i in range(1):
            self.trigPub.publish(str(self.params))
        #rospy.Subscriber("/naoqi_camera/imageSequences", String, self.processor)
        #rospy.spin()
        cachedImagesAddress = "imagesCache/"
        
        time.sleep(1) #give some time for data to be saved on disk
        if self.mode ==0:            
            self.stackImages(cachedImagesAddress)
            #self.predictObjects()
        
    
    
    def stackImages(self, rootAddress):
        for f in os.listdir(rootAddress):
            img = PILImage.open(rootAddress + f)
            img = img.resize((self.width,self.height))
            img = np.array(img)
            #if self.stackedImages is None:
            stackedImages = self.stackedImages.append(img)
        
    
    def processor(self, data):
        image = np.asarray(eval(data.data))
        
        predictedLabels =self.predictObjects(image)
        

        
        
        
    def predictObjects(self):

        #img_path = 'elephant.jpg'
        #img = image.load_img(img_path, target_size=(224, 224))
        #x = image#.img_to_array(image)
        #print("CCCCCCCCCCCCCCCCCCCCC")
        #print(imageAddress)
        ##img = PILImage.open(imageAddress)
        ##img = img.resize((224,224))
        ##img.show()
        ##x = np.asarray(img)
        #print(x.shape)
        ##x = np.expand_dims(x, axis=0)
        
        x = self.stackedImages
        x = np.asanyarray(x)
        print("BBBBBBBBBBBBBBB")
        print(x.shape)
        x = preprocess_input(x)
        preds = self.modelImage.predict(x)
        decodedPreds = decode_predictions(preds, top=3)
        print('Predicted:', decodedPreds)

        print("predicting Objects")
        
        objNames = []
        for image in decodedPreds:
            for obj in image:
                objNames.append(obj[1].replace('_', ' '))
        
        self.modelImage=None
        
        return objNames
        
        
        
    def showImage(self, numpyImage):
        toimage(numpyImage).show()

       
        
        
        
if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--pip", type=str, default="10.0.0.3",
                        help="Robot IP address.  On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--pport", type=int, default=9559,
                        help="Naoqi port number")

    
    args = parser.parse_args()
    pip = args.pip
    pport = args.pport
    '''
    rospy.init_node("imageProcessor")
    imgP = imageProcessor()
    print(imgP.self.predictObjects())
    
    
