'''
@author: voroujak
'''

import rospy
from std_msgs.msg import String

from imageProcessor import *
from SeMaCDialog import *
from AIMLEngine import *

import gc

class Portal:
    def __init__(self):
        # the transcript that has been detected by ASR
        self.speechHypothesis = None
        
        # create two answer engine
        self.AIMLEng = AIMLEngine()
        self.SeMaCEng = SeMaCDialog()
        
        # publish answers and let the robot TTS it
        self.robotReplyTopic = rospy.Publisher("/naoqi_TTS/replyBack", String, queue_size = 10)
        
        # listen to the best transcript came from ASR
        rospy.Subscriber("/naoqi_ASR/bestSpeechHypothesis", String, self.getASR)
        
        rospy.spin()



    def preprocessSentence(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.replace("cannot", "can not")
        sentence = sentence.replace("'re", " are")
        sentence = sentence.replace("'d", " would")
        sentence = sentence.replace("n't", " not")
        sentence = sentence.replace("'", " ")
        return sentence 
        
        
    def getASR(self, data):
        self.speechHypothesis = data.data
        
        self.speechHypothesis = self.preprocessSentence(self.speechHypothesis)
        reply = None

        print ("HUMAN: " + str(self.speechHypothesis))
        
        imageModelLoaded = False
        # triggering camera. #TODO make a better way of triggering camera
        if ("you see" in self.speechHypothesis) or ("percept" in self.speechHypothesis):
            imageModelLoaded = True
            #flashing memory. there are some problem with using two NN model into one gpu. memory issue and "tensor does not belong to this graph". currently at each change of NN, it flash memory and reload the NN models again.
            gc.collect()
            perception = imageProcessor()
            #perception.model._make_predict_function()
            detectedObjects = perception.predictObjects()
            reply = "i see " + str(detectedObjects[0]) + ' and ' +  str(detectedObjects[1]) + ' and ' +str(detectedObjects[2])
            
            
        #if not(reply == None):
        else:
            #try to get the answer from SeMaC, if not possible, get it from AIML
            #try:
            gc.collect()
            reply = self.SeMaCEng.reply(self.speechHypothesis)
            if reply == None:
                reply = self.AIMLEng.AIMLAnswer(self.speechHypothesis)
            #except Exception:
                #reply = self.AIMLEng.AIMLAnswer(self.speechHypothesis)
            
        print ("ROBOT : " +  reply)
        
        self.replyToRobot(reply)
        
        # flash memory and load semacDialog
        if imageModelLoaded == True:
            self.SeMaCEng.reinitModel()
            imageModelLoaded == False 
    
    # publish the answer to the robot-reply-back-topic        
    def replyToRobot(self, reply):    
        self.robotReplyTopic.publish(reply)
        

        



#starting point of portal
if __name__ == "__main__":

    rospy.init_node("MainPortal")
    Portal()
