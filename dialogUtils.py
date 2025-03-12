"""
@author: voroujak
"""


from keras.models import load_model
import pickle as pk
import keras.backend as K
from nltk import word_tokenize
import numpy as np
import nltk
from keras.utils import normalize

from scipy.spatial import distance


max_len=15
predThresholdFE = 0.1
predThresholdFT = 0.5


def f1FT(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true= y_true[: , : , :len(frameTypes)+1]
    y_pred = y_pred[ : , : , :len(frameTypes)+1]
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def f1FE(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true= y_true[: , : , len(frameTypes)+1+1:-1]
    y_pred = y_pred[ : , : , len(frameTypes)+1+1:-1]
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def f1BothLabelss(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_trueFT= y_true[: , : , :len(frameTypes)+1]
    y_trueFE= y_true[: , : , len(frameTypes)+1+1:-1]
        
    y_predFT = y_pred[ : , : , :len(frameTypes)+1]
    y_predFE = y_pred[ : , : , len(frameTypes)+1+1:-1] #len(frameTypes)+1+1+len(labels)+1

    '''
    precisionFT = precision(y_trueFT, y_predFT)
    precisionFE = precision(y_trueFT, y_predFT)

    precision = (precisionFT + precisionFE)/2
    
    recallFT = recall(y_trueFE, y_predFE)
    recallFE = recall(y_trueFE, y_predFE)

    recall = (recallFT+recallFE)/2
    '''
    y_true = K.concatenate((y_trueFT, y_trueFE), axis=2)
    y_pred = K.concatenate((y_predFT, y_predFE), axis=2)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def zeroPadding (X):
    Xframe = np.zeros(shape=(len(X),max_len, 300))
    for sentenceIndex in range(len(X)):
        Xframe[sentenceIndex, :len(X[sentenceIndex]), :] = X[sentenceIndex]
    return Xframe


def embedder (word):
    indexWord = wordS.tolist().index(word)
    return embeddingMatrix[indexWord, 1:]

def sentencePrep(sentence):
    Xss = []
    tokens = word_tokenize(sentence)

    for i in range(len(tokens)):
        surfaceWord = tokens[i]
        #surfaceWordPOS = nltkPosTagger([surfaceWord])[0][1]

        #print(surfaceWord)
        #print(surfaceWordPOS)
        #sentenceX.append([surfaceWord, surfaceWordPOS])
        #print(surfaceWord)
        Xss.append(embedder(surfaceWord.lower()))#(vocabs.index(surfaceWord.lower()), allPOS.index(surfaceWordPOS)))
        #Xss.append([surfaceWord, surfaceWordPOS])
    return Xss



def testSentence (model, testSent):
    result = []
    resultGeneral = []
    #FEINDEXES = []
    #FTINDEXES = []
    sentenceFE = []
    sentenceFT = []
    testX = sentencePrep(testSent)
    testX = zeroPadding([testX])
    #testX = pad_sequences(maxlen=max_len, sequences=[testX], padding="post", value=0)#len(vocabs)+1)
    
    senTokens = nltk.word_tokenize(testSent)

    testX = normalize(testX)


    #testFrameTypes,  testP = model.predict(testX)
    test = model.predict(testX)
    testFrameTypes = test[:,:, :len(frameTypes)+1+1]
    testP = test[:,:,len(frameTypes)+1+1 : ] #for FEs
    labelIndexes = np.argmax(testP , axis=-1)
    frameIndexes = np.argmax(testFrameTypes, axis = -1)
    
    FEIndexSplicit = np.argmax(test[:,:, len(frameTypes)+1+1:-2], axis =-1)
    FTIndexSplicit = np.argmax(test[:,:, :len(frameTypes)], axis = -1)
    
    sentenceFEs = []
    sentenceFTs = []
    for wordIndex in range(len(senTokens)):
        if testP[0][wordIndex][FEIndexSplicit[0][wordIndex]] > predThresholdFE:
            wordFE = labels[FEIndexSplicit[0][wordIndex]]
            wordFEConfidence = testP[0][wordIndex][FEIndexSplicit[0][wordIndex]]
        else:
            wordFE= 'None'
            wordFEConfidence = 'None'
            
        if testFrameTypes[0][wordIndex][FTIndexSplicit[0][wordIndex]] > predThresholdFT:
            wordFT = frameTypes[FTIndexSplicit[0][wordIndex]]
            wordFTConfidence = testFrameTypes[0][wordIndex][FTIndexSplicit[0][wordIndex]]
        else:
            wordFT= 'None'
            wordFTConfidence = 'None'
        sentenceFEs.append([wordFE,wordFEConfidence])
        sentenceFTs.append([wordFT,wordFTConfidence])

    return sentenceFEs, sentenceFTs

def predictionAnalyser(FEs, FTs, sentence):
    #what is object
    #are we confident or not
    
    #what is the intent of sentence
    FTsCountingList = [0]*len(frameTypes)
    for wordIndex in range(len(FTs)):
        if (FTs[wordIndex][0] == None) or (FTs[wordIndex][0] == 'None'):
            continue
        FTsCountingList[frameTypes.index(FTs[wordIndex][0])] += 1
    sentenceIntent = frameTypes[np.argmax(FTsCountingList)]
    
    #what is the entity that the user is talking about
    entities = ['Object', 'Entity', 'Theme', 'Object ', 'Towards ']
    entity= []
    sentenceVec = sentence.split()
    for wordIndex in range(len(FEs)):
        if FEs[wordIndex][0] in entities:
            entity.append(sentenceVec[wordIndex])
    
    #Now finding assignment, what is the assignment to entity
    assignment = []
    for wordIndex in range(len(FEs)):
        if FEs[wordIndex][0] == None:
            continue
        
        if sentenceIntent == 'Labeling':
            if FEs[wordIndex][0] == 'Label':
                
                assignment.append(sentenceVec[wordIndex])
        if sentenceIntent == 'Possession':
            if ('LU' in FEs[wordIndex][0]) and (sentenceVec[wordIndex] in ['her', 'his', 'she', 'their', 'its', 'my', 'mine', 'yours', 'your']):
                assignment.append(sentenceVec[wordIndex])
            if FEs[wordIndex][0] == 'Owner':
                assignment.append(sentenceVec[wordIndex])
        if sentenceIntent == 'BeingLocated':
            if ('Location' in FEs[wordIndex][0]) or ('PrepositionPlace' in FEs[wordIndex][0]):
                assignment.append(sentenceVec[wordIndex])
        if sentenceIntent == 'HavingOrLackingAccess':
            if 'LU' in FEs[wordIndex][0]:
                assignment.append(sentenceVec[wordIndex])
        if sentenceIntent == 'Weight':
            if ('LU' in FEs[wordIndex][0]) or ('Value' in FEs[wordIndex][0]):
                assignment.append(sentenceVec[wordIndex])
        if sentenceIntent == 'Size':            
            if 'LU' in FEs[wordIndex][0]:
                assignment.append(sentenceVec[wordIndex])
        if sentenceIntent == 'Being_operational':
            if 'LU' in FEs[wordIndex][0]:
                assignment.append(sentenceVec[wordIndex])
            
    #polarity Finder
    reverseMeaning = False
    if ('not' in sentence) or ('n\'t' in sentence):
        reverseMeaning = True
        
            
    return sentenceIntent, entity, assignment, reverseMeaning


#Pronoun reverser, you to I, I to you
def pronounReverser(entity, assignmentM):
    for assignmentIndex in range(len(assignmentM)):
        if assignmentM[assignmentIndex] in ['me','my', 'i','mine']:
            assignmentM[assignmentIndex] = 'you'
        else:
            if assignmentM[assignmentIndex] in ['you', 'your']:
                assignmentM[assignmentIndex] = 'mine'
                
    for entityIndex in range(len(entity)):
        if entity[entityIndex] in ['my', 'i','mine']:
            entity[entityIndex] = 'yours'
        else:
            if entity[entityIndex] in ['you', 'your']:
                entity[entityIndex] = 'mine'

    return entity, assignmentM

def sentenceGenerator(FEs, FTs, sentenceIntent, entity, assignmentM, polarity):

    entity, assignmentM = pronounReverser(entity, assignmentM)

    FEs, FTs, sentenceIntent, entity, assignmentM, polarity = FEs, FTs, sentenceIntent, entity, assignmentM, polarity
    
    #there should be at least two word with label (FT and FE), otherwise should not generate any sentence
    validFEs = 0
    validFTs = 0
    for fElement in FEs:
        if fElement[0] != 'None' :
            validFEs += 1
    for fElement in FTs:
        if fElement[0] != 'None' :
            validFTs += 1
    #print(validFEs)
    if (validFEs <2) or (validFTs <2):
        return None
    
    def labelSentGen():
        sent= 'init'
        if not polarity:
            sent = 'Ok, ' + ' '.join(word for word in entity) + ' I save it as '+ ' '.join(word for word in assignmentM)
        if polarity:
            sent = 'oh, ok, ' + ' '.join(word for word in entity) + ' is  not '+ ' '.join(word for word in assignmentM)
        return sent

    def ownershipSentGen():
        sent= 'init'
        if not polarity:
            sent = 'Got it! ' + ' '.join(word for word in entity) + ' belongs to '+ ' '.join(word for word in assignmentM)
        if polarity:
            sent = 'oh, ok, ' + ' '.join(word for word in entity) + ' does not belongs to '+ ' '.join(word for word in assignmentM)
        return sent

    def positionSentGen():
        sent= 'init'
        if not polarity:
            sent = 'I see, ' + ' '.join(word for word in entity) + ' is '+ ' '.join(word for word in assignmentM)
        if polarity:
            sent = 'oh, ok, ' + ' '.join(word for word in entity) + ' is  not '+ ' '.join(word for word in assignmentM)
        return sent

    def restrictionsSentGen():
        sent= 'init'
        if assignmentM[0] == 'good':
            assignmentSemantic = True
        else:
            assignmentSemantic = False 
        semantic = (assignmentSemantic and (not polarity)) or ((not assignmentSemantic) and (polarity))
        if semantic:
            sent = 'oh, ok, I save ' + ' '.join(word for word in entity) + ' as a NO zone.'
        if (not semantic):
            sent = 'Ok, good, I save ' + ' '.join(word for word in entity) + ' with credential access.'
        return sent

    def functionalitySentGen():
        sent= 'init'
        #print('THIS IS TTTTTT')
        if assignmentM[0] == 'good':
            assignmentSemantic = True
        else:
            assignmentSemantic = False 
            
        semantic = (assignmentSemantic and (not polarity)) or ((not assignmentSemantic) and (polarity))

        if (not semantic):
            sent = 'oh, ok, I save ' + ' '.join(word for word in entity) + ' as improper.'
        if semantic:
            sent = 'Ok, it seems ' + ' '.join(word for word in entity) + ' is still working'
        return sent

    def weightSentGen():
        sent= 'init'
        if not polarity:
            sent = 'oh, ok, ' + ' '.join(word for word in entity) + ' is '+ ' '.join(word for word in assignmentM)
        if polarity:
            sent = 'oh, ok, ' + ' '.join(word for word in entity) + ' is  not '+ ' '.join(word for word in assignmentM) + ' then'
        return sent

    def sizeSentGen():
        sent= 'init'
        if not polarity:
            sent = 'ok, got it, ' + ' '.join(word for word in entity) + ' is '+ ' '.join(word for word in assignmentM)
        if polarity:
            sent = 'oh, ok, ' + ' '.join(word for word in entity) + ' is  not '+ ' '.join(word for word in assignmentM) + ' then'
        return sent

    
    if sentenceIntent == 'Labeling':  
        sent = labelSentGen()
        
    elif sentenceIntent == 'Possession':
        sent = ownershipSentGen()
        
    elif sentenceIntent == 'BeingLocated':
        sent = positionSentGen()
                    
    elif sentenceIntent == 'HavingOrLackingAccess':
        sent = restrictionsSentGen()
    
    elif sentenceIntent == 'Weight':
        sent = weightSentGen()
            
    elif sentenceIntent == 'Size':            
        sent = sizeSentGen()
        
    elif sentenceIntent == 'Being_operational':
        #print ('THISI S ISSSSSSSSSSSSSS')
        sent = functionalitySentGen()
    else:
        sent = None #'I could not understand the sentence'
       

    
        
    return sent
        


def prediction(sentence):
    return testSentence(model, sentence)



def distanceFinder(word1, word2):
    vec1 = np.asarray(embedder(word1),dtype=float)
    vec2 = np.asarray(embedder(word2), dtype=float)
    return distance.cosine(vec1,vec2)

def distanceFinderVec ( vectorWord1, word2):
    vec1 = np.asarray(vectorWord1, dtype=float)
    vec2 = np.asarray(embedder(word2), dtype=float)
    return distance.cosine(vec1,vec2)


def LUSynthesizer (vecOfLexicalUnits, FT):
    # these two are for functionality, and restrictions
    good = ['good', 'yes', 'true', 'existance' ]
    bad = ['bad', 'no', 'false', 'not']
    
    sizeLevels = ['huge', 'big', 'small', 'tiny']
    weightLevels = ['feather', 'massive']
    
    if (FT == 'Being_operational') or (FT == 'HavingOrLackingAccess'):
        goodDists = 0
        badDists = 0
        embeddedWords =[0]*300
        for lIndex in range(len(vecOfLexicalUnits)):
            embeddedWords += embedder(vecOfLexicalUnits[lIndex])
        for i in range(len(good)):
            luword = embeddedWords
            goodDists += distanceFinderVec(luword ,good[i])
            badDists += distanceFinderVec(luword, bad[i])
        if goodDists < badDists:
            return ['good']
        else:
            return ['bad']
    
    if (FT == 'Size'):
        sizeDists = [5]*len(sizeLevels)
        embeddedWords =[0]*300

        for lIndex in range(len(vecOfLexicalUnits)):
            embeddedWords += embedder(vecOfLexicalUnits[lIndex])
        for sizeLevelIndex in range(len(sizeLevels)):
            sizeDists[sizeLevelIndex] = distanceFinderVec(embeddedWords,sizeLevels[sizeLevelIndex])
        closestSizeLevelIndex = np.argmin(np.asarray(sizeDists))
        return [sizeLevels[closestSizeLevelIndex]]
    
    if (FT == 'Weight'):
        weightDists = [5]*len(weightLevels)
        embeddedWords =[0]*300
        for lIndex in range(len(vecOfLexicalUnits)):
            embeddedWords += embedder(vecOfLexicalUnits[lIndex])
        for weightLevelIndex in range(len(weightLevels)):
            weightDists[weightLevelIndex] = distanceFinderVec(embeddedWords,weightLevels[weightLevelIndex])
        closestWeightLevelIndex = np.argmin(np.asarray(weightDists))
        return [weightLevels[closestWeightLevelIndex]]
    #other cases
    if FT in ['Possession', 'Labeling', 'BeingLocated']:
        return vecOfLexicalUnits
            
#K.clear_session()
def reinitModel():
    global model
    #model = None
    K.clear_session()
    model = load_model('/home/voroujak/Main/Sapienza/MS/Thesis/checkingModel.h5', custom_objects={'f1FE': f1FE, 'f1FT':f1FT , 'f1BothLabelss': f1BothLabelss})
    model._make_predict_function()
    
filehandlerFELabels = open("/home/voroujak/Main/Sapienza/MS/Thesis/preProcesses/FELabels","rb")
labels = pk.load(filehandlerFELabels)
    
filehandlerFTLabels = open("/home/voroujak/Main/Sapienza/MS/Thesis/preProcesses/FTLabels","rb")
frameTypes = pk.load(filehandlerFTLabels)
    
filehandlerwordS = open("/home/voroujak/Main/Sapienza/MS/Thesis/preProcesses/vocabulary", "rb")
wordS = pk.load(filehandlerwordS)
    
filehandlerEmbedding = open("/home/voroujak/Main/Sapienza/MS/Thesis/preProcesses/vocabularyArrays", "rb")
embeddingMatrix = pk.load(filehandlerEmbedding)
global model
model = load_model('/home/voroujak/Main/Sapienza/MS/Thesis/checkingModel.h5', custom_objects={'f1FE': f1FE, 'f1FT':f1FT , 'f1BothLabelss': f1BothLabelss})
model._make_predict_function()

print('ALL LOADED!')
