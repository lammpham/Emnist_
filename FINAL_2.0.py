# Importiere benötigte Bibliotheken
import sklearn.datasets as skl_data
import sklearn.neural_network as skl_nn
import pickle
import matplotlib.pyplot as plt
from skimage import io
from picamera import PiCamera
import cv2
import numpy

mlp = pickle.load(open("MLP_classifier_200",'rb'))
i = 1

camera = PiCamera()


while i < 10:
    inp = input("Press Enter to predict Letter!\nOr press Q to quit.\nTo see the Camera view, press C.")
    if inp == 'Q':
        break
    if inp == 'C':
        camera.start_preview(fullscreen=False, window=(200,200,800,800))
        
    #print(inp)
    camera.capture(f'Letter.jpg')
    print("Captured Letter")

    file = r'/home/pi/Desktop/Final/Letter.jpg'
    handwritten = []
    
    gray_image= cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    (tresh, blackwhite_image) = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY)
    
    handwritten.append(blackwhite_image)
    typestory = ""
    typestory2 = ""
    
    cropped_image = blackwhite_image[200:900, 500:1200]
    
    for img_resize in handwritten:
        img_resized = cv2.resize(cropped_image, (28,28), interpolation=cv2.INTER_LINEAR)
        img_resized = cv2.bitwise_not(img_resized)

    plt.imshow(cropped_image, cmap='gray')
    cv2.imwrite('Letter.jpg', img_resized)
    #plt.show()

    # INFERENZ auf RASPBERRY
    # Inferenz mit MLP Modell

    #print('Versuchen wir es mit dem Bild')
    test_digit = io.imread('Letter.jpg')
    #test_digit_prediction = mlp.predict(test_digit.reshape(1,784))
    single_item_array = (numpy.array(test_digit)).reshape(1,784)
    prediction = mlp.predict(single_item_array)
    typestory = typestory + str(chr(prediction[0]+64))
    typestory2 = typestory2 + str(chr(prediction[0]+96))
    #labels_txt=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    
    #print(f'test_digit_prediction: {test_digit_prediction[0]}')
    #if prediction in range(prediction[0]+64):
    print("Predicted Capital Letter:",typestory)
    #if prediction in range(prediction[0]+96):
    print("Predicted Lowercase Letter:",typestory2)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    ax.imshow(test_digit.reshape(28,28), cmap='gray')
    ax.axis('off')
    plt.show()