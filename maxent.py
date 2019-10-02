# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from corpus import Document, NamesCorpus, ReviewCorpus
from random import shuffle
import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier


class MaxEnt(Classifier):

    def __init__(self):
        super(MaxEnt)
        self.maxlength = 0
        self.dict = {}
        self.classifier = None
        self.model = {}

    # def get_model(self): return {}
    #
    # def set_model(self, model): return model
    #
    # model = property(get_model, set_model)

    def getFeatures(self,instances):
        feature_matrix = []
        labels = []
        if isinstance(instances, NamesCorpus):
            instances = instances[:self.maxlength]
            shuffle(instances)
            self.model.mapping = {0: 'female', 1: 'male'}
            for i in instances:
                i.feature_vector = i.features()
                feature_matrix.append(i.feature_vector)
                if i.label == 'female':
                    labels.append(1)
                else:
                    labels.append(0)
        elif isinstance(instances, ReviewCorpus):
            instances = instances[:self.maxlength]
            shuffle(instances)
            self.model['mapping'] = {-1: 'negative', 0: 'neutral', 1: 'positive'}
            for i in instances:
                i.feature_vector = i.features()
                feature_matrix.append(i.data)
                if i.label == 'positive':
                    labels.append(1)
                elif i.label == 'neutral':
                    labels.append(0)
                else:
                    labels.append(-1)
            feature_matrix = np.array(feature_matrix)
            if i.feature_vector == 'bigram':
                count_vect = CountVectorizer(ngram_range=(1, 2))
            elif i.feature_vector == 'bagofwords':
                count_vect = CountVectorizer()
            else:
                count_vect = CountVectorizer()
            X_train_counts = count_vect.fit_transform(feature_matrix)
            tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
            feature_matrix = tf_transformer.transform(X_train_counts)
            labels = np.array(labels)
            self.dict['tf_transformer'] = tf_transformer
            self.dict['count_vect'] = count_vect
        else:
            pass
        return feature_matrix,labels

    def train(self, instances, maxlength, batch_size, l2_value, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        ##Get Features
        self.maxlength = maxlength
        if l2_value>1 :
            l2_value = 1
        X,Y = self.getFeatures(instances)
        print("featurization finished")
        trainSize = math.floor(maxlength* 0.8)
        trainX, testX = (X[:trainSize], X[trainSize:])
        trainY, testY = (Y[:trainSize], Y[trainSize:])
        self.train_sgd(trainX, trainY, 0.0001, batch_size, l2_value, trainSize)
        predY = self.classifier.predict(testX)
        return accuracy_score(testY, predY)

    def generateBatch(self, X, Y, batch_size, trainSize):
        count = 0
        while count < trainSize:
            end = count+batch_size
            if count+batch_size > trainSize:
                end = trainSize
            batchX = X[count:end]
            batchY = Y[count:end]
            yield batchX, batchY
            count += batch_size

    def train_sgd(self, X, Y, learning_rate, batch_size, l, maxlength):
        """Train MaxEnt model with Mini-batch Stochastic Gradient 
        """
        mini_batch = self.generateBatch(X, Y, batch_size, maxlength)
        classifier = SGDClassifier(loss="log",
                                   penalty="l1",
                                   l1_ratio=l,
                                   learning_rate='optimal',
                                   tol=0.0002)
        for batchX, batchY in mini_batch:
            classifier.partial_fit(batchX, batchY, classes=np.unique(Y))
        print("train finished")
        self.model['dict'] = self.dict
        self.classifier = classifier
        self.model['classifier'] = classifier

    def classify(self, instance):
        if len(self.model) > 0 and isinstance(instance,str):
            vect = self.model['dict']['count_vect']
            clf = self.model['classifier']
            transformer = self.model['dict']['tf_transformer']
            formatpred = np.array([instance])
            pred = vect.transform(formatpred)
            pred_tf = transformer.transform(pred)
            res = clf.predict(pred_tf)[0]
            return self.model['mapping'][res]


