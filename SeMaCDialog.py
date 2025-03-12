"""
@author: voroujak
"""

import dialogUtils


class SeMaCDialog:
    def __init__(self):
        print("SEMAC INITIALIZED")
    def reply(self,sentence):
        #dialogUtils.reinitModel()
        
        # predict FEs and FTs by NN, and filter them based on prbabilities.
        FEs, FTs = dialogUtils.prediction(sentence)
        
        #given FEs and FTs of each sentence, find intent of sentence, entities and properties that are assigned to entities(assignment), and finding if the sentence has positive polarity or negative (if there is any NOT or not), which might change the meaning.
        sentenceIntent, entity, assignment, polarity = dialogUtils.predictionAnalyser(FEs, FTs, sentence)
        
        # clustering assignet properties to predefined clusters.
        assignmentMeaning = dialogUtils.LUSynthesizer(assignment, sentenceIntent)

        # generate sentence
        generatedSentence = dialogUtils.sentenceGenerator(FEs, FTs, sentenceIntent, entity, assignmentMeaning, polarity)
        
        return generatedSentence
     
    # re-initializing model. Flashing memory and load the model, and make the model as default graph.   
    def reinitModel(self):
        dialogUtils.reinitModel()
        #print(generatedSentence)
        

