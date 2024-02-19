import numpy as np
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import tkinter as tk
from PIL import Image, ImageDraw
import os

# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование изображений в одномерные массивы и нормализация значений
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# Преобразование меток в формат one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Создание модели нейросети
model = Sequential()

# Добавление первого скрытого слоя с 16 нейронами
model.add(Dense(128, input_dim=784, activation='relu'))

# Добавление второго скрытого слоя с 16 нейронами
model.add(Dense(64, activation='relu'))

# Добавление третьнго скрытого слоя с 16 нейронами
model.add(Dense(32, activation='relu'))

# Добавление выходного слоя с 10 нейронами (по числу классов) и функцией активации softmax
model.add(Dense(10, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Оценка точности модели на тестовой выборке
scores = model.evaluate(X_test, y_test, verbose=0)
print(f"Точность на тестовой выборке: {scores[1]*100:.2f}%")



class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Drawing App")

        self.canvas = tk.Canvas(self.master, width=280, height=280, bg="black", cursor="cross")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw_brush)

        self.button_save = tk.Button(self.master, text="GOOOO", command=self.go)
        self.button_save.pack()

        self.button_clear = tk.Button(self.master, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.button_close = tk.Button(self.master, text="Close", command=self.close_window)
        self.button_close.pack()
        self.drawing = Image.new("L", (280, 280), color="black")  # Создаем изображение для сохранения
        self.draw = ImageDraw.Draw(self.drawing)

    def draw_brush(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="white", outline="white")

        # Рисуем на изображении
        self.draw.ellipse([x-10, y-10, x+10, y+10], fill="white")

    def go(self):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(project_dir, "test.png")
        if file_path:
            self.drawing.save(file_path)
            print(f"Image saved at: {file_path}")
            # Загрузка и преобразование изображения
            image = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))
            image = image / 255.0
            image = image.reshape((1, 784))

            # Предсказание класса на изображении
            prediction = np.argmax(model.predict(image), axis =-1)

            print("ваша цифра: ", prediction)
    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing = Image.new("L", (280, 280), color="black")
        self.draw = ImageDraw.Draw(self.drawing)

    def close_window(self):
        self.master.destroy()

# Создание экземпляра приложения
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()