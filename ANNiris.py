from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import neural_network
from sklearn.metrics import confusion_matrix


class ANNiris:

    def __init__(self):
        self.inputs, self.outputs, self.outputNames = self.loadIrisData()
        self.trainInputs, self.trainOutputs, self.testInputs, self.testOutputs = self.splitData(self.inputs, self.outputs)
        self.trainInputs, self.testInputs = self.normalisation(self.trainInputs, self.testInputs)

        self.classifier = neural_network.MLPClassifier()

    # step1: Incarcarea datelor - setul de date iris (3 clase)
    def loadIrisData(self):
        data = load_iris()
        inputs = data['data']
        outputs = data['target']
        outputNames = data['target_names']
        featureNames = list(data['feature_names'])
        feature1 = [feat[featureNames.index('sepal length (cm)')] for feat in inputs]
        feature2 = [feat[featureNames.index('petal length (cm)')] for feat in inputs]
        inputs = [[feat[featureNames.index('sepal length (cm)')], feat[featureNames.index('petal length (cm)')]] for
                  feat in inputs]
        return inputs, outputs, outputNames

    # step2: Impartirea datelor in date de antrenament si date de test
    #        20% - date de test
    #        80% - date de antrenament
    def splitData(self, inputs, outputs):
        np.random.seed(5)
        indexes = [i for i in range(len(inputs))]
        trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
        testSample = [i for i in indexes if not i in trainSample]

        trainInputs = [inputs[i] for i in trainSample]
        trainOutputs = [outputs[i] for i in trainSample]
        testInputs = [inputs[i] for i in testSample]
        testOutputs = [outputs[i] for i in testSample]

        return trainInputs, trainOutputs, testInputs, testOutputs

    # step3 : normalizare datelor de antrenament si a datelor de test
    def normalisation(self, trainData, testData):
        scaler = StandardScaler()
        if not isinstance(trainData[0], list):
            # encode each sample into a list
            trainData = [[d] for d in trainData]
            testData = [[d] for d in testData]

            scaler.fit(trainData)  # fit only on training data
            normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
            normalisedTestData = scaler.transform(testData)  # apply same transformation to test data

            # decode from list to raw values
            normalisedTrainData = [el[0] for el in normalisedTrainData]
            normalisedTestData = [el[0] for el in normalisedTestData]
        else:
            scaler.fit(trainData)  # fit only on training data
            normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
            normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
        return normalisedTrainData, normalisedTestData

    # step4 : Invatare model (cu tool neural_network.MLPClassifier())
    #         Training the classifier, identify (by training) the classification model
    def training(self):
        self.classifier.fit(self.trainInputs, self.trainOutputs)

    # step5 : Testare model
    #         Predictii pentru datele de test
    def classification(self):
        computedTestOutputs = self.classifier.predict(self.testInputs)
        return computedTestOutputs

    # step6: calcul metrici de performanta (acc)
    def eval(self, realLabels, computedLabels, labelNames):
        confMatrix = confusion_matrix(realLabels, computedLabels)
        acc = sum([confMatrix[i][i] for i in range(len(labelNames))]) / len(realLabels)
        precision = {}
        recall = {}
        for i in range(len(labelNames)):
            precision[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[j][i] for j in range(len(labelNames))])
            recall[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[i][j] for j in range(len(labelNames))])
        return acc, precision, recall, confMatrix

