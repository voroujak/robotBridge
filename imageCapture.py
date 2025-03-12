'''
@author: voroujak
'''

#shall be run by python 2, naoqi
import rospy
from naoqi import ALProxy
from PIL import Image as PILImage
from sensor_msgs.msg import Image as Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge, CvBridgeError
#import vision_definitions
import numpy as np
import json
import argparse

class imageCapture:

    def __init__(self, quality=2, colorSpace=11, nuImagesSequence = 2, 
            savePath = "imagesCache/", fps = 5, PIP = "127.0.0.1", PPORT=9559):
        self.quality = quality # between 0 and 3 included
        self.colorSpace = colorSpace
        self.nuImagesSequence = nuImagesSequence
        self.savePath = savePath
        self.fps = fps
        self.IP = PIP
        self.PORT = PPORT

        #be careful, I am sending images one by one to queue, it might be better to stack them all in an array, then send it.
        #it seems sending images into topic is time consuming, better to save it on hdd
        #self.imagePublisher = rospy.Publisher('/naoqi_camera/imageSequences', String, queue_size=self.nuImagesSequence)
         

        

        self.camProxy = ALProxy("ALVideoDevice", self.IP, self.PORT) 
        #self.videoClient = self.camProxy.subscribe("rospy_gvm", 0,self.quality, self,colorSpace, self.fps)
        self.camClient = self.camProxy.subscribe("naoqiCamera", self.quality, self.colorSpace, self.fps) 
              
        #try:
        for self.iteration in range(self.nuImagesSequence):
            self.captureImage()
        #except Exception:
        #    pass
        self.camProxy.unsubscribe(self.camClient)
            
        
    def captureImage(self):
        #get the image
        self.img = self.camProxy.getImageRemote(self.camClient)

        self.imgWidth = self.img[0]
        self.imgHeight = self.img[1]
        self.imgData = self.img[6]

        self.im = PILImage.frombytes("RGB", (self.imgWidth, self.imgHeight), self.imgData)
        
        #changing image type to string, easier for rosTopic between python3 and python2
        self.stringedImage  = json.dumps(np.asarray(self.im).tolist())

        #for sending image message, because of using python3 on the other side, this might now work well, instead sending a String message then transforming it might do the job as well.
        #self.imgMessage = CvBridge().cv2_to_imgmsg(self.im)
        #Deprecated since I found difficulties in sending a huge String like image
        #self.imagePublisher.publish(self.stringedImage)

        #Saving PIL image, useful for saving and loading, better to use ROS publisher/subscriber, if possible later!
        self.im.save(self.savePath + str(self.iteration) + ".png", "PNG")
        
        
def cameraTrigger(data):
    params = eval(data.data)
    print(data.data)
    imageCapture(quality=int(params["quality"]), 
        colorSpace=int(params["colorSpace"]), 
        nuImagesSequence = int(params["nuImagesSequence"]), 
        savePath = str(params["savePath"]), 
        fps = int(params["fps"]),
        PIP = pip,
        PPORT = pport)
        
        
        
if __name__ == "__main__":
    global pip
    global pport
    parser = argparse.ArgumentParser()
    parser.add_argument("--pip", type=str, default="10.0.0.5",
                        help="Robot IP address.  On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--pport", type=int, default=9559,
                        help="Naoqi port number")

    
    args = parser.parse_args()
    pip = args.pip
    pport = args.pport


    rospy.init_node("imageCapture")
    rospy.Subscriber("/naoqi_camera/captureTrigger", String, cameraTrigger, queue_size=1)
    print("initializing")
    rospy.spin()
    #imageCapture()
