import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Create a button to open an image file
        self.file_button = tk.Button(self, text='Open Image', command=self.open_file)
        self.file_button.pack()

        # Create a label to display the image
        self.image_label = tk.Label(self)
        self.image_label.pack()

        # Create a label to display the model's prediction
        self.prediction_label = tk.Label(self)
        self.prediction_label.pack()

        # Load the model
        self.model = tf.keras.models.load_model('model.h5')

    def open_file(self):
        # Open a file dialog to choose an image file
        filepath = filedialog.askopenfilename()

        # Open the image and resize it to 128x128
        image = Image.open(filepath)
        image = image.resize((128, 128))

        # Convert the image to a numpy array and preprocess it for the model
        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make a prediction using the model
        predictions = self.model.predict(image_array)
        prediction = np.argmax(predictions[0])

        # Update the image and prediction labels
        self.image_label.configure(image=ImageTk.PhotoImage(image))
        self.prediction_label.configure(text=f'Prediction: {prediction}')

if __name__ == '__main__':
    app = App()
    app.mainloop()