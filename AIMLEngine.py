'''
@author: voroujak
'''

import aiml
import os
class AIMLEngine:
    def __init__(self, AIMLSet =  "/home/voroujak/datasets/botdata/alice"):
        self.k = aiml.Kernel()
        self.learn(self.k , "/home/voroujak/datasets/botdata/alice")
        print("AIML INITIALIZED")
        
    def learn(self, interpreter, path):
        for root, directories, file_names in os.walk(path):
            for filename in file_names:
                if filename.endswith('.aiml'):
                    interpreter.learn(os.path.join(root, filename))
        
        print ('Number of categories: ' + str(interpreter.numCategories()))
        
    def AIMLAnswer(self, sentence):
        return (self.k.respond(sentence))
