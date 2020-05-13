import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import neural_network
from sklearn.metrics import confusion_matrix

class ANN:
    def __init__(self, inputs, outputs):
        self.inputs, self.outputs = inputs, outputs
        self.trainInputs, self.trainOutputs, self.testInputs, self.testOutputs = self.splitData(self.inputs, self.outputs)
        self.trainInputs = [self.flatten(el) for el in self.trainInputs]
        self.testInputs = [self.flatten(el) for el in self.testInputs]

        self.classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5, ), activation='relu', max_iter=100, solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)

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

        trainInputs = np.array(trainInputs)
        trainOutputs = np.array(trainOutputs)
        testInputs = np.array(testInputs)
        testOutputs = np.array(testOutputs)

        return trainInputs, trainOutputs, testInputs, testOutputs

    # check if the data is uniform distributed over classes
    def flatten(self, mat):
        x = []
        for line in mat:
            for el in line:
                for k in el:
                    x.append(k)
        return x

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
    def eval(self, computedLabels):
        realLabels = np.array(self.testOutputs)
        labelNames = ["CuFiltru" ,"FaraFiltru"]
        confMatrix = confusion_matrix(realLabels, computedLabels)
        acc = sum([confMatrix[i][i] for i in range(len(labelNames))]) / len(realLabels)
        precision = {}
        recall = {}
        for i in range(len(labelNames)):
            precision[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[j][i] for j in range(len(labelNames))])
            recall[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[i][j] for j in range(len(labelNames))])
        return acc, precision, recall, confMatrix

    def getTest(self):

        return self.trainInputs