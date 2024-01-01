import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from sklearn import (datasets,model_selection as skms, neighbors, metrics as mt, linear_model as lm)
import sqlite3, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps




def loadData():
    mnist = fetch_openml('mnist_784', version = 1, as_frame = False)
    X,y = mnist.data, mnist.target 
    return X,y

def showValues():
    instancia = 0
    some_digit = X[instancia]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap="binary")
    plt.axis("off")
    plt.title('Imagen que representa un {}'.format(y[instancia])) 
    plt.show()

    fig, ax = plt.subplots(10,10, figsize = (4, 4))
    instancia = 0
    for fila in range(10):
        for columna in range(10):
            ax[fila,columna].imshow(X[instancia].reshape(28,28), cmap = 'binary')
            ax[fila, columna].axis('off')
            instancia+=1
        #plt.pause(0.1)
def train_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = neighbors.KNeighborsClassifier()
    model.fit(X_train, y_train)
    return model 
    
def predict_number(image,knn_model):
    img = image.resize((28,28))
    img_invert = ImageOps.invert(img)
    grey_matrix= np.array(img_invert.convert('L'))
    vector_784 = grey_matrix.reshape((1,784))
    prediction = knn_model.predict(vector_784)
    return prediction 

def drawNumber():
    root = tk.Tk()
    root.title("Dibujar un número")

    canvas_size = 280
    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
    canvas.pack()

    image_size = 28
    image = Image.new("L", (canvas_size, canvas_size), color="white")
    image = image.resize((image_size, image_size))

    scale_factor = canvas_size / image_size

    draw = ImageDraw.Draw(image)

    def paint(event):
        x1, y1 = (event.x - 1) / scale_factor, (event.y - 1) / scale_factor
        x2, y2 = (event.x + 1) / scale_factor, (event.y + 1) / scale_factor
        canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
        draw.line([x1, y1, x2, y2], fill="black", width=2)

    def finish_drawing():
        root.quit()

    canvas.bind("<B1-Motion>", paint)

    done_button = tk.Button(root, text="Listo", command=finish_drawing)
    done_button.pack()

    root.mainloop()

    # Crea una nueva variable para almacenar la imagen redimensionada antes de que termine el bucle
    image_resized = image.resize((image_size, image_size))

    return image_resized



X,y =loadData()

model = train_model(X,y)

banner = 0

while banner<5:
    image = drawNumber()
    value=predict_number(image, model)
    
    image_array = np.array(image)
    plt.imshow(image_array, cmap="gray_r")
    plt.show()
    print(value)
    banner += 1