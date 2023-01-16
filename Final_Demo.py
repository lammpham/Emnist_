# Importiere benötigte Bibliotheken
import sklearn.datasets as skl_data
import sklearn.neural_network as skl_nn
import pickle
import matplotlib.pyplot as plt
from skimage import i
from picamera import PiCamera

camera = PiCamera()
i = 1

while True:
    inp = input("Press Enter to Capture Letter!")
    print(inp)
    camera.capture(f'Letter.jpg')
    print("Captured Letter")
    print(i)
    i = i+1

# INFERENZ auf RASPBERRY
# Trainiertes Netzwerk wieder einlesen
mlp = pickle.load(open("MLP_CLASS", 'rb'))

# Inferenz mit MLP Modell
print('Versuchen wir es mit einem Bild')
test_digit = io.imread('Letter.jpg')
test_digit_prediction = mlp.predict(test_digit.reshape(1,784))
print("Predicted value",test_digit_prediction)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
ax.imshow(test_digit.reshape(28,28), cmap='gray')
ax.axis('off')
plt.show()