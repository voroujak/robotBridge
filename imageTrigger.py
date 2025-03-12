import rospy
import json

from std_msgs.msg import String

rospy.init_node('camera_triggerer')

params = {'quality':2, 'colorSpace':11, 'nuImagesSequence' : 1, 'savePath' : "imagesCache/", 'fps' : 5}


trigPub = rospy.Publisher("/naoqi_camera/captureTrigger", String, queue_size=1)
for i in range(50000):
    trigPub.publish(str(params))
