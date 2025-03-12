'''
@author: voroujak
'''

import qi
import argparse
import sys
import os
import rospy
from std_msgs.msg import String
#from naoqi_bridge_msgs.msg import EventStamped

'''
--event should be AlSpeechRecognition/Status
or a better one: SpeechDetected
values can be :
Idle, ListenOn, SpeechDetected, EndOfProcess, ListenOff, Stop
'''
pub = None
def onEventSpeechDetected(value):
    pubSpeechDetected.publish(str(value))
    print "valueSpeechDetected=",value
    
def onEventTTS(value):
    pubTTS.publish(str(value[1]))
    print "valueTTS= ", value[1]

def main():
    global pubSpeechDetected
    global pubTTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--pip", type=str, default="10.0.0.3",#os.environ['PEPPER_IP'],
                        help="Robot IP address.  On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--pport", type=int, default=9559,
                        help="Naoqi port number")
    
    args = parser.parse_args()
    pip = args.pip
    pport = args.pport
    eventSpeechDetected = "SpeechDetected"
    eventTTS = "ALTextToSpeech/Status"


    
    #rospy.Rate(100)
    pubSpeechDetected = rospy.Publisher('~/naoqi_ASR/SpeechDetected', String, queue_size=10)
    
    pubTTS = rospy.Publisher("/naoqi_TTS/Status", String, queue_size = 10)
    

    ''' subscribe to : ALTextToSpeech/Status
        when ever "done" is detected, publish True to a rostopic, so it start processing
    '''


    #Starting application
    try:
        connection_url = "tcp://" + pip + ":" + str(pport)
        app = qi.Application(["ReactToTouch", "--qi-url=" + connection_url ])
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + pip + "\" on port " + str(pport) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    app.start()
    session = app.session

    #Starting services
    memory_service  = session.service("ALMemory")
      
    #subscribe to any change on SpeechDetected
    subscriberSpeechDetected = memory_service.subscriber(eventSpeechDetected)
    idEventSpeechDetected = subscriberSpeechDetected.signal.connect(onEventSpeechDetected)
    
    #subscribe to any change on TTS
    subscriberTTS = memory_service.subscriber(eventTTS)
    idEventTTS = subscriberTTS.signal.connect(onEventTTS)
    
    print "subscribed to events!"
    
    #Program stays at this point until we stop it
    app.run()

    #Disconnecting callbacks
    subscriberSpeechDetected.signal.disconnect(idEventSpeechDetected)
    subscriberTTS.signal.disconnect(idEventTTS)
    
    print "Finished"


if __name__ == "__main__":
    #initializing rosNode
    rospy.init_node('naoqi_eventSubscriber')
    main()
