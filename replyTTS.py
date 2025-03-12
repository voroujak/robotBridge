'''
@author: voroujak
'''

from naoqi import ALProxy
from std_msgs.msg import String
import rospy
import argparse

class naoqiTTS:
    def __init__(self, PIP , PPORT, SPEED):
        self.PIP = PIP
        self.PPORT = PPORT
        self.tts = ALProxy("ALTextToSpeech", self.PIP, self.PPORT)
        rospy.Subscriber("/naoqi_TTS/replyBack", String, self.playTTS)
        print("TTS is running!")
        rospy.spin()
        
        
    def playTTS(self, data):
        sentence = data.data
        self.tts.say(sentence)
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pip", type=str, default="127.0.0.1",
                        help="Robot IP address.  On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--pport", type=int, default=43225,
                        help="Naoqi port number")
    parser.add_argument("--speed", type=int, default=100,
                        help="speed")
    
    args = parser.parse_args()
    pip = args.pip
    pport = args.pport
    speed = args.speed

    rospy.init_node("naoqi_tts")
    naoqiTTS(PIP = pip, PPORT = pport, SPEED= speed)
        
