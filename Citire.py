from sklearn.datasets import load_sample_images
import glob
from PIL import Image
import numpy as np

class Citire:

    def incarcareDate(self):
        outputs = []
        inputs = []
        for f in glob.iglob("C:/Users/Florina/PycharmProjects/ia_lab10/pozeCuSepia/*"):
            inputs.append(np.asarray(Image.open(f)))
            outputs.append(1)

        for f in glob.iglob("C:/Users/Florina/PycharmProjects/ia_lab10/pozeFaraSepia/*"):
            inputs.append(np.asarray(Image.open(f)))
            outputs.append(0)

        inputs = np.array(inputs)
        outputs = np.array(outputs)

        noData = len(inputs)
        permutation = np.random.permutation(noData)
        inputs = inputs[permutation]
        outputs = outputs[permutation]

        return inputs, outputs
