import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        #NEED TO CHANGE THESE
        
        # Create a button to open an image file
        self.file_button = tk.Button(self, text='Open Image', command=self.open_file)
        self.file_button.pack()

        self.image_label = tk.Label(self)
        self.image_label.pack()
        
        # Create a label to display the image
        self.canvas = tk.Canvas(self, width=224, height=224)
        self.canvas.pack()

        # Create a label to display the model's prediction
        self.prediction_label = tk.Label(self)
        self.prediction_label.pack()

        # Load the model
        self.model = tf.keras.models.load_model('C:/Users/tomke/Downloads/model/bt_ai_test2-67acc-1lss-20-03-23.h5', compile=False)

    def open_file(self):
        filepath = filedialog.askopenfilename()

        image = Image.open(filepath)
        image = image.resize((224, 224))

        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = self.model.predict(image_array)
        prediction = np.argmax(prediction,axis=1)[0]

        #ImageShow.show(image)
        #self.canvas.delete('all')
        #photo = ImageTk.PhotoImage(image)
        #print(photo)
        #self.canvas.create_image(0, 0, anchor='nw', image=photo)
        #self.image_label.configure(image=photo)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        self.prediction_label.configure(text=f'Prediction: {prediction}')
        

        if prediction == 0:
          print("glioma identified")
          self.prediction_label.configure(text=f'glioma identified')
        if prediction == 1:
          print("meningioma identified")
          self.prediction_label.configure(text=f'meningioma identified')
        if prediction == 2:
          print("No tumor identified")
          self.prediction_label.configure(text=f'no tumor identified')
        if prediction == 3:
          print("pituitary identified")
          self.prediction_label.configure(text=f'pituitary identified')

if __name__ == '__main__':
    app = App()
    app.mainloop()
