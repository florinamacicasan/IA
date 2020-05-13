'''
    1.Crearea unei baze cu imagini (cu si fara filtru sepia) si etichetele corespunzatoare
    2. Antrenarea unui clasificator pentru clasificarea imaginilor
    cu si fara filtru
    3. Testarea clasificatorului
'''

from Citire import Citire
from ANN import ANN

def main():
    '''
        Citire date
    '''
    citire = Citire()
    inputs, outputs = citire.incarcareDate()

    clasificator = ANN(inputs, outputs)

    # Invatare model
    clasificator.training()

    # Testare model
    predictii = clasificator.classification()

    # Calcul metrici de performanta pentru model (acuratete, precizie, rapel, matriceDeConfuzie)
    acuratete, precizie, rapel, matriceConfuzie = clasificator.eval(predictii)
    print("Acuratete = ", acuratete)
    print("Precizie = ", precizie)
    print("Recall = ", rapel)


main()
