from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import sys

A = "C:/Users/NEW.PC/Desktop/image.jpg"


# Open Image and return Nd Array
def OpenImage(link):
    img = Image.open(link)
    img = img.convert('1')
    img = np.asarray(img)
    return img


# Reshape image from X*X to X+2 * X+2
def Reshape_Image(image):
    width, height = image.shape[0], image.shape[0]
    X_image = np.zeros((width + 2, height + 2))
    for i in range(height):
        for j in range(width):
            X_image[i + 1, j + 1] = image[i, j]
    for i in range(514):
        X_image[0, i] = X_image[1, i]
        X_image[513, i] = X_image[512, i]
    for i in range(1, 514):
        X_image[i, 513] = X_image[i, 512]
        X_image[i, 0] = X_image[i, 1]
    return X_image


# Test function that Reshape_Image  is working properly
def Equality_test(X_image):
    x = X_image[512, 1:513]
    y = X_image[513, 1:513]
    if (x == y).all():
        print("True")
    else:
        print(" not equal 2")
    a = X_image[1:513, 0]
    b = X_image[1:513, 1]
    if (a == b).all():
        print("True again")
    else:
        print(" not equal 2")


# implement Xor MLP
def unit_step(v):
    if v >= 0:
        return 1
    else:
        return 0


def perceptron(x, w, b):
    v = np.dot(w, x) + b
    y = unit_step(v)
    return y


def NOT_percep(x):
    return perceptron(x, w=-1, b=0.5)


def AND_percep(x):
    w = np.array([1, 1])
    b = -1.5
    return perceptron(x, w, b)


def OR_percep(x):
    w = np.array([1, 1])
    b = -0.5
    return perceptron(x, w, b)


def XOR_net(x):
    gate_1 = AND_percep(x)
    gate_2 = NOT_percep(gate_1)
    gate_3 = OR_percep(x)
    new_x = np.array([gate_2, gate_3])
    output = AND_percep(new_x)
    return output


def X3_X5_filtre(image, i, j):
    x, y = 0, 0
    for i_ in range(i - 2, i + 3):
        for j_ in range(j - 2, j + 2):
            if i - 1 <= i_ <= i + 1 and j - 1 <= j_ <= j + 1:
                x += image[i_, j_]
            y += image[i_, j_]
    if x >= 5:
        x = 1
    else:
        x = 0
    if y >= 13:
        y = 1
    else:
        y = 0
    return x, y

def traitment():
    image = OpenImage(A)
    width, height = image.shape[0], image.shape[1]
    image = Reshape_Image(image)
    sh1 = np.zeros((width, height))
    sh2 = np.zeros((width, height))
    # loop in the image and select X*X neighborgs
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            x, y = X3_X5_filtre(image, i, j)
            input = [x, y]
            result = XOR_net(input)
            if result == 0:
                # To sh1
                sh1[i, j] = image[i, j]
            else:
                # to sh2
                sh2[i, j] = image[i, j]
    plt.imshow(image, cmap='Greys')
    plt.show()
    # plt.imshow(sh1, cmap='Greys')
    # plt.show()
    # plt.imshow(sh2, cmap='Greys')
    # plt.show()
    # plt.imsave('C:\\Users\lenovo\Desktop\sh2_2_1.png', sh1, cmap=cm.gray)
    # plt.imsave('C:\\Users\lenovo\Desktop\sh2_2_2.png', sh2, cmap=cm.gray)


