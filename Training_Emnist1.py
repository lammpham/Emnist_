import os
# import cv2
#import numpy
import matplotlib.pyplot as plt
from emnist import extract_training_samples
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pickle

importfromweb = False
# Holen die Daten aus OpenML Webseit
if importfromweb:
    x, y = extract_training_samples('letters')
    # Optional: Benutze picle um Daten und Labels auf Platte zu speichern oder zu laden
    pickle.dump(x,open("emnist_data",'wb'))
    pickle.dump(y,open("emnist_labels",'wb'))
else:
    x = pickle.load(open("emnist_data", 'rb'))
    y = pickle.load(open("emnist_labels", 'rb'))
    
x = x / 255.

x_train, x_test = x[:60000], x[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

# Sichern, dass die Daten korrekt heruntergeladen wurden
img_index = 14000
img = x_train[img_index]
print("Image Label: " + str(chr(y_train[img_index]+96)))
plt.imshow(img.reshape((28,28)))


# 2.MLP mit mehreren Hidden-Layers
mlp2 = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,), max_iter=60, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
                     learning_rate_init=.1)


mlp2.fit(x_train, y_train)

y_pred = mlp2.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm)


print("Training set score: %f" % mlp2.score(x_train, y_train))
print("Test set score: %f" % mlp2.score(x_test, y_test))
# Speichere Netzwerk auf der Platte
pickle.dump(mlp2, open("MLP_classifier", 'wb'))
