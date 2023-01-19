# Importiere benötigte Bibliotheken
import sklearn.datasets as skl_data
import sklearn.neural_network as skl_nn
import pickle
import matplotlib.pyplot as plt
from skimage import io
from picamera import PiCamera
import cv2
import numpy

#Trainiertes Netzwerk wieder einlesen
mlp = pickle.load(open("MLP_classifier_200",'rb'))

#Instantiierung der Kamera
camera = PiCamera()
#Eigentlich unnötiges i
i = 1
#Main Schleife für das Programm
while i < 10:
    #Input Abfrage für die verschiedenen Optionen:
    #ENTER = Bild aufnehmen und Vermutung aufstellen lassen
    #C = Kamerabild wird eingeschaltet
    #Q = Bricht die Main while Schleife
    inp = input("Press Enter to predict Letter!\nOr press Q to quit.\nTo see the Camera view, press C.")
    if inp == 'Q':
        break
    if inp == 'C':
        camera.start_preview(fullscreen=False, window=(200,200,800,800))
    # Aufnahme eines Bildes als "Letter.jpg"
    camera.capture(f'Letter.jpg')
    print("Captured Letter")
    file = r'/home/pi/Desktop/Final/Letter.jpg'
    
    handwritten = []
    
    #Letter.jpg wird Grau skaliert
    gray_image= cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    #Grau skaliertes Letter.jpg wird durch ein Threshold auf Schwarz/Weiß bearbeitet
    (tresh, blackwhite_image) = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY)
    
   
    handwritten.append(blackwhite_image)
    typestory = ""
    typestory2 = ""
    
    #Hier wird in demSchwarz/Weiß Letter.jpg die Mitte aus geschnitten
    cropped_image = blackwhite_image[200:900, 500:1200]
    
    #Ausgeschnittenes Bild wird für EMNIST passend auf 28x28 Pixel skaliert
    for img_resize in handwritten:
        img_resized = cv2.resize(cropped_image, (28,28), interpolation=cv2.INTER_LINEAR)
        img_resized = cv2.bitwise_not(img_resized)
    
    #Anzeigen des bearbeitetes Bildes und speichern auf Letter.jpg
    plt.imshow(cropped_image, cmap='gray')
    cv2.imwrite('Letter.jpg', img_resized)

    # Inferenz mit MLP Modell
    test_digit = io.imread('Letter.jpg')
    single_item_array = (numpy.array(test_digit)).reshape(1,784)
    prediction = mlp.predict(single_item_array)
    #Großbuchstaben
    typestory = typestory + str(chr(prediction[0]+64))
    #Kleinbuchstaben
    typestory2 = typestory2 + str(chr(prediction[0]+96))
    
    #Hier wird die Vermutung des NNs ausgegeben
    print("Predicted Capital Letter:",typestory)
    print("Predicted Lowercase Letter:",typestory2)
    
    #Zeigt das End bearbeitete Bild an
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    ax.imshow(test_digit.reshape(28,28), cmap='gray')
    ax.axis('off')
    plt.show()
