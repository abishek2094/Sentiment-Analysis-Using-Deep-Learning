import pandas
import numpy as np
import re
import nltk.data
import warnings
import sys
import os.path as ospath
from gensim.models import word2vec
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

"""
    In this assignment, I decided to understand the different approaches to feature creation and extraction and what role
    it plays in the accuracy of the sentiment classifier on the Movie review data set procured from Kaggle available at
    "https://www.kaggle.com/c/word2vec-nlp-tutorial/data". This code was only written to gain familiarity and understanding,
    the credits go to the individulas who procurred the data and implemented the libraries used in this code as well as to 
    the hosts who made available the tutorial/data.

    The assignment flow at a high level is as follows :

    1) Use all the reviews provided along with the Word2vec library in order to train the word vectors.
    2) Try different approaches to feature creation such as :
        2.1) Averaging/ Mean of all word vectors for each review.
        2.2) Root mean square of all word vectors for each review.
        2.3) Standard deviation of all word vectors for each review
        2.4) Clustering all the words and using cluster count in each reciew as feature.
    3) Once the features are available, we use a classification technique, in this case a random forest classifier to 
    classify sentiment of the reviews as Good or Bad.

    Detailed explanation :-

        1) Training the word2vec model:
            The data source at "https://www.kaggle.com/c/word2vec-nlp-tutorial/data" provides three basic data sets of
            movie reviews - labeledTrainData.tsv which is a lablelled set of 25000 reviews as well as their sentiment,
            testData.tsv which is an unlabled set of 25000 reviews and unlabeledTrainData.tsv which is also an unlabelled
            set of 50000 reviews. I have used all of these sets to train the word2vec model since more the data, the better
            the word vectors. This is done after cleaning the reviews such as removing punctuations, URLs etc.

        2) Exploring different approaches to Feature creation:
            To study the accuracy, I have chosen to use only the labeledTrainData.tsv file which is a lablelled set of 
            25000 reviews as well as their sentiment. I have split this set into 22000 records of training data and the
            remaining 3000 records will be used as my test data.

            The approaches I have tried for feature creation are as follows : 

                2.1) Averaging/ Mean of all word vectors for each review.
                    Each review has many words. We average out the word vectors for each word to create the feature vector
                    for each review.

                2.2) Root mean square of all word vectors for each review.
                    Each review has many words. We compute the Root mean square of word vectors for each word to create 
                    the feature vector for each review.

                2.3) Standard deviation of all word vectors for each review
                    Each review has many words. We compute the Standard deviation of word vectors for each word to create 
                    the feature vector for each review.

                2.4) Clustering all the words and using cluster count in each reciew as feature.
                    Here, we use K-means clustering to cluster all the words in the vocabulary of the trained word2vec
                    model. Features for each review are a vector of how many words from each of these clusters belong to
                    the review.

            3) Once the features are available, we use a classification technique, in this case a random forest classifier to 
                classify sentiment of the reviews as Good or Bad. We train the classifier using the 22000 labelled records
                and test accuracy of each feature creation method on the 3000 test data.


"""


def printProgress(percent, message=""):
    """A sinmple function to print progress for time consuming operations"""
    sys.stdout.write(
        "\r\t" + message + "\t[" + "*" * round(percent / 10) + " " * (10 - round(percent / 10)) + "]  " + str(
            int(percent)) + "% Completed.")


def readDataFromFileIntoDataFrame(file):
    """A sinmple function to read the data from the mentioned file into a dataframe using pandas"""
    dataFrame = None
    if "tsv" in file:
        dataFrame = pandas.read_csv(file, header=0, delimiter="\t", quoting=3)

    elif "csv" in file:
        dataFrame = pandas.read_csv(file, header=0, quoting=3)

    if "id" in dataFrame.columns:
        del dataFrame["id"]

    return dataFrame


def cleanReviewSentences(sentence, removeStopwords=False, stopWords=None):
    """
        A function that takes a review string and cleans it by removing HTML formatting, any non alpha-numeric character
        , convert to lower case and if requested remove stop words respectively. This function returns a list of clean words.
    """

    cleanedSentence = (BeautifulSoup(sentence)).get_text()

    cleanedSentence = re.sub("[^a-zA-Z0-9 ]", " ", cleanedSentence)

    cleanedWords = (cleanedSentence.lower()).split()

    cleanedWords = [word for word in cleanedWords if len(word) > 0]

    if removeStopwords == True:
        cleanedWords = [word for word in cleanedWords if not word in stopWords]

    return cleanedWords


def preprossesReviews(review, tokenizer):
    """
        This function cleans the reviews by removing any URLs from it, then splitting the review into sentence and then
        cleaning each sentence by calling "cleanReviewSentences". This function returns a list of all cleaned words in the
        review.
    """

    cleanedListOfSentences = []

    review = re.sub("http:[a-zA-Z0-9/\.]*", " ", review)

    reviewSentences = tokenizer.tokenize(review.strip())

    for sentence in reviewSentences:
        if len(sentence) > 0:
            cleanedListOfSentences.append(cleanReviewSentences(sentence))

    return cleanedListOfSentences


def prepareTrainingData(reviews, tokenizer):
    """
        This function is used to clean all the reviews to be used for training the word2vec model. This function calls
        "preprossesReviews" for each review.
    """
    cleanedListOfSentences = []
    i = 0
    dataSize = len(reviews)
    for review in reviews:
        if i % 100 == 0:
            printProgress(((i + 1) * 100) / dataSize, "Preparing data for training word2vec ...")
        cleanedListOfSentences += preprossesReviews(review, tokenizer)
        i += 1
    printProgress(100)

    return cleanedListOfSentences


def setUpDataForWordVectors(tokenizer):
    """
        This function reads data from labeledTrainData.tsv, unlabeledTrainData.tsv and testData.tsv and accumalates reviews
        from all the three to be used for training the word2vec model.
    """
    trainingData = []

    dataFrame1 = readDataFromFileIntoDataFrame(
        "G:\\PythonWorkspace\\Pycharm\\Word2VecFinal\\Data\\labeledTrainData.tsv")
    trainingData += prepareTrainingData(dataFrame1['review'], tokenizer)

    dataFrame2 = readDataFromFileIntoDataFrame(
        "G:\\PythonWorkspace\\Pycharm\\Word2VecFinal\\Data\\unlabeledTrainData.tsv")
    trainingData += prepareTrainingData(dataFrame2['review'], tokenizer)

    dataFrame3 = readDataFromFileIntoDataFrame("G:\\PythonWorkspace\\Pycharm\\Word2VecFinal\\Data\\testData.tsv")
    trainingData += prepareTrainingData(dataFrame3['review'], tokenizer)

    return trainingData


def wordVectorTraining(numberOfFeatures=300):
    """
        This function is used to train the word2vec model or load an already trained model if it exists.
    """
    modelName = "wordVectormodel"
    if ospath.isfile("wordVectormodel") == True:
        model = word2vec.Word2Vec.load(modelName)
    else:
        minWordCount = 40
        numberOfWorkers = 4
        context = 10
        downSampling = 1e-3

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        wordVecTrainingData = setUpDataForWordVectors(tokenizer)

        model = word2vec.Word2Vec(wordVecTrainingData, workers=numberOfWorkers, size=numberOfFeatures,
                                  min_count=minWordCount, window=context, sample=downSampling)
        model.init_sims(replace=True)

        model.save(modelName)

    return model


def partitionTrainingSet(trainingSet):
    """
        This function is used to partition the data set into training and test sets as described
    """
    trainSet = trainingSet[:22000]
    testSet = trainingSet[22000:]

    return trainSet, testSet


def createFeatureFromReviewWordVectors(reviewWords, vocabulary, model, numberOfFeatures):
    """
        This is a very important function that creates the feature vectors by using mean, root mean square or standard
        deviation of all the word vectors in each review.
    """

    featureVector = np.zeros((1, numberOfFeatures), dtype=np.float32)[0]

    numberOfWords = 0

    if type == 1 or type == 3:
        """This is feature creation using mean of all word vectors which gave a test data accuracy of 83.1%"""
        for word in reviewWords:
            if word in vocabulary:
                numberOfWords = numberOfWords + 1.
                featureVector = np.add(featureVector, model[word])

        featureVector = np.divide(featureVector, numberOfWords)

    if type == 2:
        """This is feature creation using root mean square of all word vectors which gave a test data accuracy of 77.4%"""
        for word in reviewWords:
            if word in vocabulary:
                numberOfWords = numberOfWords + 1.
                featureVector = np.add(featureVector, np.square(model[word]))

        featureVector = np.divide(featureVector, numberOfWords)
        featureVector = np.sqrt(featureVector)

    if type == 3:
        """This is feature creation using standard deviation of all word vectors which gave a test data accuracy of 77%"""
        featureVec = np.zeros((1, numberOfFeatures), dtype=np.float32)
        for word in reviewWords:
            if word in vocabulary:
                featureVec = np.add(featureVec, np.square(np.subtract(model[word], featureVector)))

        featureVector = np.sqrt(np.divide(featureVec, numberOfWords))

    return featureVector


def createFeatureForReviews(reviews, model, numberOfFeatures, type):
    """This function calls def createFeatureFromReviewWordVectors for each review in the training set"""
    reviewNumber = 0
    numberOfReviews = len(reviews)
    featureVectors = np.zeros((numberOfReviews, numberOfFeatures), dtype=np.float32)
    vocabulary = set(model.wv.index2word)

    for reviewWords in reviews:
        if reviewNumber % 100 == 0:
            printProgress(((reviewNumber + 1) * 100) / numberOfReviews, "Creating Feature Vectors for reviews ..")

        featureVectors[reviewNumber] = createFeatureFromReviewWordVectors(reviewWords, vocabulary, model,
                                                                          numberOfFeatures, type)
        reviewNumber += 1
    printProgress(100, "Creating Feature Vectors for reviews ..")

    return featureVectors


def setupFeatures(trainingData, testData, model, numberOfFeatures=300, type=1):
    """This function sets up the vectors for the training as well as the test data"""
    stopWords = set(stopwords.words("english"))

    cleanReviews = []

    for review in trainingData["review"]:
        review = re.sub("http:[a-zA-Z0-9/\.]*", " ", review)
        cleanReviews.append(cleanReviewSentences(review, True, stopWords))

    trainingDataVectors = createFeatureForReviews(cleanReviews, model, numberOfFeatures, type)

    cleanTestData = []

    for review in testData["review"]:
        review = re.sub("http:[a-zA-Z0-9/\.]*", " ", review)
        cleanTestData.append(cleanReviewSentences(review, True, stopWords))

    testDataVectors = createFeatureForReviews(cleanTestData, model, numberOfFeatures, type)

    return trainingDataVectors, testDataVectors


def trainRandomForest(trainingDataVectors, trainingData, numberOfTrees=100):
    """This function is used to train the random forest usin the created features"""
    printProgress(0, "Creating Random forest Classifier .....")
    randomForestClassifier = RandomForestClassifier(n_estimators=numberOfTrees)
    randomForestClassifier = randomForestClassifier.fit(trainingDataVectors, trainingData["sentiment"])
    printProgress(100, "Creating Random forest Classifier .....")

    return randomForestClassifier


def predictResults(testDataVectors, testData, randomForestClassifier):
    """This function classifies the sentiment on the test data using the training random forest model and returns accuracy"""
    predictonResult = randomForestClassifier.predict(testDataVectors)

    numberOfCorrectPredictions = sum(np.equal(predictonResult, testData["sentiment"]) == True)

    numberOfTestData = len(testData)

    return ((float(numberOfCorrectPredictions) / numberOfTestData) * 100)


def clusterWords(model, numberOfWordsPerCluster=5):
    """This function clusters all the words of the word2vec model vocabulary and returns a dictionary of cluster assignments"""

    wordVectors = model.wv.syn0
    numberOfClusters = int(model.wv.syn0.shape[0] / numberOfWordsPerCluster)

    kmeansClusteringObject = KMeans(n_clusters=numberOfClusters)

    clusterAssignments = kmeansClusteringObject.fit_predict(wordVectors)

    clusterAssignments = dict(zip(model.wv.index2word, clusterAssignments))

    return clusterAssignments


def createFeaturesWithClusters(reviewWords, clusterAssignments, numberOfClusters):
    """
        This function creates the feature vector of the passed review by computing the number of words in the review
        belonging to each cluster.
    """
    featureVector = np.zeros((1, numberOfClusters), dtype=np.float32)[0]

    for word in reviewWords:
        if word in clusterAssignments:
            index = clusterAssignments[word]
            featureVector[index] += 1

    return featureVector


def createFeatureForReviewsUsingClusters(dataFrame, clusterAssignments):
    """
        This function first cleans all the reviews and calls createFeaturesWithClusters for each review to create the feature matrix
        This approach resulted in the best accuracy of 85.1% on the test set.
    """
    stopWords = set(stopwords.words("english"))
    cleanReviews = []

    for review in dataFrame["review"]:
        review = re.sub("http:[a-zA-Z0-9/\.]*", " ", review)
        cleanReviews.append(cleanReviewSentences(review, True, stopWords))

    reviewNumber = 0
    numberOfReviews = len(cleanReviews)

    numberOfClusters = max(clusterAssignments.values()) + 1

    featureVectors = np.zeros((numberOfReviews, numberOfClusters), dtype=np.float32)

    for reviewWords in cleanReviews:
        if reviewNumber % 100 == 0:
            printProgress(((reviewNumber + 1) * 100) / numberOfReviews, "Creating Feature Vectors for reviews ..")

        featureVectors[reviewNumber] = createFeaturesWithClusters(reviewWords, clusterAssignments, numberOfClusters)
        reviewNumber += 1
    printProgress(100, "Creating Feature Vectors for reviews ..")

    return featureVectors


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    model = wordVectorTraining(300)

    labelledData = readDataFromFileIntoDataFrame(
        "G:\\PythonWorkspace\\Pycharm\\Word2VecFinal\\Data\\labeledTrainData.tsv")

    actualTrainData, actualTestData = partitionTrainingSet(labelledData)

    # type = 1 (Word Vector mean), 2 (Word Vector root mean square), 3 (Word Vector standard deviation), 4 (Word Vector clustering)
    type = 1

    if type <= 3:
        actualTrainDataVectors, actualTestDataVectors = setupFeatures(actualTrainData, actualTestData, model, 300, type)

    else:
        clusterAssignments = clusterWords(model, 5)
        actualTrainDataVectors = createFeatureForReviewsUsingClusters(actualTrainData, clusterAssignments)
        actualTestDataVectors = createFeatureForReviewsUsingClusters(actualTestData, clusterAssignments)

    randomForestClassifier = trainRandomForest(actualTrainDataVectors, actualTrainData, 100)
    accuracy = predictResults(actualTestDataVectors, actualTestData, randomForestClassifier)

    print("\n\n\nTest Set accuracy : ", accuracy, "%")