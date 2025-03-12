###
#
# @author: voroujak
#
###
source ~/anaconda3/etc/profile.d/conda.sh
#activating pyhton 2.7, naoqi
conda activate naoqi

#running replyback of robot. nao TTS service
xterm -hold -e "python replyTTS.py --pip '10.0.0.3' --pport 9559"  &

#running Image capture of nao
xterm -hold -e "python imageCapture.py --pip '10.0.0.3' --pport 9559" &

#running a dumb ASR @ only on real robot
xterm -hold -e "python asr/asr.py --pip '10.0.0.3' --pport 9559" &

#running microphone stream @ only on real robot ##need naoqi_bridge naoqi_sensors_py
xterm -hold -e "roslaunch naoqi_sensors_py microphone.launch nao_ip:=10.0.0.3" &

#running event subscribe @ only on real robot
xterm -hold -e "python eventSubscriber.py --pip '10.0.0.3' --pport 9559" &

#activating python3, tf
conda activate tf 

#running android ASR, shall be used without ASR, microphone_launch, and eventSubscriber
##xterm -hold -e "python android_interface_node.py" &

xterm -hold -e "python main.py" &

xterm -hold -e "python remoteASR.py" &


