import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
print(mnist.keys()) #DESCR - Opis zbioru danyhc, data - tablica (wiersz-przykładm kolumna-cecha), target - tablica etykiet
# print(mnist.DESCR)
# print(mnist.data)

X, y = mnist['data'], mnist['target']
print("X.shape:", X.shape)
print("y.shape:", y.shape)

#Przykładowa cyfra
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0] #5
some_digit_image = some_digit.reshape(28, 28) #przekształcenie w macierz o rozmiarze 28x28
plt.imshow(some_digit_image, cmap='binary')
plt.axis('off')
plt.show()
print("some_digit:", y[0] )

y = y.astype(np.uint8) # etykieta jest łańuchem znkaów, więc trzeba trzeba przekształcić na liczbę całkowitą

#Podział na zbiór uczący i testowy
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#UCZENIE KLASYFIKATORA BINARNEGO
#identyfikacja tylko cyfry 5
y_train_5 = (y_train == 5) #TRUE dla piaek
y_test_5 = (y_test == 5)

#1 - Klasyfikator stochastycznego spadku wzdłuż gradientu (SGD)
