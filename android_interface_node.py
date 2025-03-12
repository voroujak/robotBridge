

'''
Inherited from lu4r project, 'Andrea Vanzo'
it needs lu4r android app, which send google ASR to socket. 
for download app, see: https://www.marrtino.org/software/app-speech
modified by @voroujak

how to run, just run it and android app. make sure both pc and android app are on the same network. aslo configure android IP and port, which its IP can be read from "ip addr show" in ubuntu, and port is fixed in params.
'''

#  Imports  #

# ROS imports
import rospy


# ROS msg and srv imports
from std_msgs.msg import String
from sensor_msgs.msg import Joy


# Python Libraries
import sys
import socket
import json


#  Variables  #
conn = None


#  Classes  #
class AndroidInterface(object):
    #  Interaction Variables  #
    reply = ''

    def __init__(self):
        global conn

        # Initialize node #
        rospy.init_node('android_interface_node', anonymous=True)
            
        # Reading params #
        self.port = int(rospy.get_param('~port', 9001))

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print ('Socket created')
            try:
                s.bind(('', self.port))
            except socket.error as msg:
                print ('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
                sys.exit()
            print ('Socket bind complete')
            s.listen(10)

            print ('Socket now listening on port ' + str(self.port))
            
            # Initialize publishers #
            #self.hypo_publisher = rospy.Publisher('/speech_hypotheses', String, queue_size=1)
            self.best_hypo_publisher = rospy.Publisher('/naoqi_ASR/bestSpeechHypothesis', String, queue_size=1)


            
            # Declare subscribers #
            #rospy.Subscriber('/joy', Joy, self.joy_callback, queue_size=10)
            #rospy.Subscriber('/blind_controller_ack', String, self.blind_controller_callback, queue_size=1)

            current_fragment = ''
            while not rospy.is_shutdown():
                print ('Waiting for connection on port ' + str(self.port))
                conn, addr = s.accept()
                print ('Connected with ' + addr[0] + ':' + str(addr[1]))
                while not rospy.is_shutdown():
                    data = conn.recv(2048)  # 512
                    data = data.decode()
                    print ('Received: ' + data)
                    if data and not data.isspace():
                        if not data:
                            continue
                        if 'REQ' in data:
                            data = data.replace('REQ', '')
                            conn.send('ACK\n'.encode())
                            continue
                        if '$' in data:
                            current_fragment = data[1:-1]
                            print ('You selected the ' + current_fragment + ' fragment')
                            continue
                        if current_fragment == 'JOY':
                            print (data)
                        elif current_fragment == 'SLU':
                            transcriptions = json.loads(data)
                            self.best_hypo = transcriptions['hypotheses'][0]['transcription']

                            self.best_hypo_publisher.publish(self.best_hypo)
                    else:
                        print ('Disconnected from ' + addr[0] + ':' + str(addr[1]))
                        break
            s.close()
        except socket.error as socket_error:
            print('Error: ', socket_error)



    #def joy_callback(self, joy_msg):
    #    print (joy_msg.buttons)
        

#  If Main  #
if __name__ == '__main__':
    AndroidInterface()

