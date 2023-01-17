# Importiere benötigte Bibliotheken
import sklearn.datasets as skl_data
import sklearn.neural_network as skl_nn
import pickle
import matplotlib.pyplot as plt
from skimage import io
from picamera import PiCamera
import cv2

mlp = pickle.load(open("MLP_CLASS",'rb'))
i = 1

camera = PiCamera()
camera.start_preview(fullscreen=False, window=(200,200,800,800))

while i < 10:
    inp = input("Press Enter to Capture Letter!\nOr press Q to quit.")
    if inp == 'Q':
        break
    print(inp)
    camera.capture(f'Letter.jpg')
    print("Captured Letter")

    file = r'/home/pi/Desktop/Projekt/Final/Letter.jpg'
    gray_image= cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    (tresh, blackwhite_image) = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    img_resized = cv2.resize(blackwhite_image, (28,28), interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.bitwise_not(img_resized)

    plt.imshow(blackwhite_image, cmap='gray')
    cv2.imwrite('Letter.jpg', img_resized)
    #plt.show()

    # INFERENZ auf RASPBERRY
    # Inferenz mit MLP Modell

    print('Versuchen wir es mit dem Bild')
    test_digit = io.imread('Letter.jpg')
    test_digit_prediction = mlp.predict(test_digit.reshape(1,784))
    labels_txt=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    print(f'test_digit_prediction: {test_digit_prediction[0]}')
    print("Predicted Letter:",labels_txt[test_digit_prediction[0]])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    ax.imshow(test_digit.reshape(28,28), cmap='gray')
    ax.axis('off')
    plt.show()