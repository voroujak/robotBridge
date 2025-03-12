'''
@author: voroujak
'''

import aiml
import os

def learn(interpreter, path):
    for root, directories, file_names in os.walk(path):
        for filename in file_names:
            if filename.endswith('.aiml'):
                interpreter.learn(os.path.join(root, filename))
    print ('Number of categories: ' + str(interpreter.numCategories()))


k = aiml.Kernel()
learn(k, "/home/voroujak/datasets/botdata/alice")

while 1:
    input_string = input('Enter a sentence: ')
    reply = k.respond(input_string)
    print(reply)
    #do_something(reply)
