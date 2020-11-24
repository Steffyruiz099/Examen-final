import cv2
import numpy as np
import math
import os
from hough import hough
from orientation_estimate import *
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time

class bandera:
    def __init__(self, image_name):
        self.image_name = image_name
        path = 'C:/Users/Steffany/Documents/Javeriana/Semestre 9/Procesamiento de Imagenes/examenfinal'
        path_file = os.path.join(path, self.image_name)
        image = cv2.imread(path_file)
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        high_thresh = 300
        bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh, L2gradient=True)
        self.hough = hough(bw_edges)


    def colores(self):

        image = np.array(self.image, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))
        intraCluster = [0, 0, 0, 0]

        for j in range(4):
            n_colors = j + 1
            t0 = time()
            image_array_sample = shuffle(image_array, random_state=0)[:10000]
            model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

            labels = model.predict(image_array)
            centers = model.cluster_centers_
            for i in range(len(labels)):
                if labels[i] == 0:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[0][0], 2) + np.power(image_array[i][1] - centers[0][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[0][2], 2)))
                elif labels[i] == 1:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[1][0], 2) + np.power(image_array[i][1] - centers[1][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[1][2], 2)))
                elif labels[i] == 2:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[2][0], 2) + np.power(image_array[i][1] - centers[2][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[2][2], 2)))
                elif labels[i] == 3:
                    intraCluster[j] += (math.sqrt(
                        np.power(image_array[i][0] - centers[3][0], 2) + np.power(image_array[i][1] - centers[3][1],
                                                                                  2) + np.power(
                            image_array[i][2] - centers[3][2], 2)))

        color = intraCluster.index(min(intraCluster)) + 1
        print("La bandera tiene: ", color," colores")

    def porcentaje(self):
        negro = np.array([0, 0, 0])
        amarillo = np.array([0, 255, 255])
        rojo = np.array([0, 0, 255])
        azul = np.array([255, 0, 0])
        blanco = np.array([255, 255, 255])
        verde = np.array([0, 255, 0])


    def orientacion(self):
        print("hola")
        accumulator = self.hough.standard_HT()

        acc_thresh = 50
        N_peaks = 11
        nhood = [25, 9]
        peaks = self.hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        [_, cols] = self.image.shape[:2]
        image_draw = np.copy(self.image)
        angulo = []
        for i in range(len(peaks)):
            rho = peaks[i][0]
            theta_ = self.hough.theta[peaks[i][1]]
            angulo.append(theta_)
            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + self.hough.center_x
            y0 = b * rho + self.hough.center_y
            c = -rho
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))

            if np.abs(theta_) < 80:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
            elif np.abs(theta_) > 100:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
            else:
                if theta_ > 0:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 0], thickness=2)
                else:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)

        print(angulo)
        if math.sin(round(angulo[0]),1) == 0 and math.sin(round(angulo[1]),1) == 0 and math.sin(round(angulo[2]),1) == 0:
            print("vertical")
        elif math.sin(round(angulo[0]),1) == 1 and math.sin(round(angulo[1]),1) == 1 and math.sin(round(angulo[2]),1) == 1:
            print("horizontal")
        else:
            print("mixto")


image_name = input('ingrese el nombre de una imagen de bandera')  # se pide el nombre de la imagen
bn = bandera(image_name)                                          # se crea un objeto en la clase
bn.colores()                                                        # se llama el método color
bn.orientacion()