'''
@author: voroujak
'''
#shall be run by python 3, tf
import rospy
import numpy as np
#import time
from scipy.io import wavfile
import collections
import math
import speech_recognition as sr
import sounddevice as sd
from logmmse import logmmse
from naoqi_bridge_msgs.msg import AudioBuffer
from std_msgs.msg import String

class remote_ASR:
    def __init__(self, bufferSeconds=0.7, packetSize=1365, samplePerSecond=16000, nuChannels = 4):
        self.buffer_time_seconds = bufferSeconds
        self.packetSize = packetSize
        self.samplePerSecond = samplePerSecond
        self.nuChannels = nuChannels
        #number of bytes for 0.7 seconds with 16kHz / each packet size
        self.max_len_buffer = math.floor((self.samplePerSecond*self.buffer_time_seconds)/self.packetSize) 
        self.audioBuf = collections.deque(maxlen=self.max_len_buffer) #creating a fixed size buffer
        self.r = sr.Recognizer()

        self.audioSegment = False
        self.segmentStarted = False
        self.TTSFinished = True
        #audioBuf = collections.deque(max_len=max_len_buffer)
        #audioSegment = None
        #self.utterance = np.asanyarray([0], dtype = np.int16)
        
        #init ros node and subscribe for callbacks, and spin it forever
        rospy.init_node('remote_ASR')

        #sending detected speech hypothesis to a topic
        self.bestSpeechHypothesisPublisher = rospy.Publisher("/naoqi_ASR/bestSpeechHypothesis", String, queue_size =10)

        rospy.Subscriber("~/naoqi_microphone/audio_raw", AudioBuffer, self.callbackBuffer)
        rospy.Subscriber("~/naoqi_ASR/SpeechDetected", String, self.callbackUtterance)
        rospy.Subscriber("/naoqi_TTS/Status", String, self.callbackTTSStatus)
        print("remote_asr is running")
        rospy.spin()

    def callbackTTSStatus(self, data):
        if data.data == "done":
            self.TTSFinished = True
        if data.data == "started":
            self.TTSFinished = False

    def callbackBuffer(self, data):
        self.MicPacket = data.data
        self.channeledPacket = np.asanyarray(self.MicPacket, dtype=np.int16).reshape((self.packetSize ,self.nuChannels))# removed transpose for test

        #stack of packets, then we open these stack and merge all together
        self.audioBuf.append(self.channeledPacket)
    

        #Note: collections stack np objects, unpack (reshape) them to original foramt
        self.reshapedAudioBuf = np.asanyarray(self.audioBuf, dtype=np.int16).reshape(-1,self.nuChannels)
    
        if self.audioSegment and self.segmentStarted: #Segment of utterance started, so put the buffer into utterance
            self.utterance = self.reshapedAudioBuf
            self.segmentStarted = False
        elif self.audioSegment and not(self.segmentStarted): # having buffer in utterance, keep continue adding packets into utterance until speech is detected 
            self.utterance = np.append(self.utterance,self.channeledPacket, axis=0)
        
  
     
    def callbackUtterance(self, data):

        if (data.data=="1") and (self.TTSFinished ==True): #Speech detected, so start recording utterance.
            self.audioSegment = True
            self.segmentStarted = True
        if data.data == "0" and (self.TTSFinished ==True): #Speech detected is finished, so ASR should do the job and then published into the topic.
            self.audioSegment = False


            self.utterance = np.mean(self.utterance, axis=1, dtype=np.int16)


            #denoising speech
            #self.utterance = logmmse(data = self.utterance, sampling_rate=16000, output_file=None, noise_threshold=0.3)
        
            # saving file as wav, for apis that need file.
            #wavfile.write("testAudiosWav/myAudio"+".wav", 16000, self.utterance)
            
            # reading saved wav file
            #with sr.AudioFile("testAudiosWav/myAudio.wav") as source:
            #    audio = r.record(source)
            
            #playing detected speech before ASR 
            sd.play(self.utterance, 16000)
            #making AudioSource object. reference: speech_recognition::AudioSource definition
            self.audio = sr.AudioData(self.utterance, 16000,2)
            #print(time.time())
            self.detectedSpeechRaw = self.r.recognize_google(self.audio)
            
            self.detectedSpeech = self.preprocessSentence(self.detectedSpeechRaw)

            self.bestSpeechHypothesisPublisher.publish(self.detectedSpeech)
            print("Google ASR:: \n ----- " + self.detectedSpeech)
            #print(time.time())
            
    #TODO remove this, as this function exist also in main.py       
    def preprocessSentence(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.replace("cannot", "can not")
        sentence = sentence.replace("'re", " are")
        sentence = sentence.replace("'d", " would")
        sentence = sentence.replace("'", " ")
        return sentence         
                      
  
if __name__ == '__main__':
    remote_ASR()
        
