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

        self.predictions_labels = []
        for i in range(4):
            label = tk.Label(self)
            label.pack()
            self.predictions_labels.append(label)
        
        # Load the model
        self.model = tf.keras.models.load_model('C:/Users/tomke/Downloads/model/bt_ai_EXP3-91acc-04lss-08-04-23.h5', compile=False)

    def open_file(self):
        filepath = filedialog.askopenfilename()

        image = Image.open(filepath)
        image = image.resize((224, 224))

        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        #prediction = self.model.predict(image_array)
        #prediction = np.argmax(prediction,axis=1)[0]

        predictions = self.model.predict(image_array)[0]
        for i in range(4):
            label = ""
            if i == 0:
                label = "Glioma confidence"
            elif i == 1:
                label = "Meningioma confidence"
            elif i == 2:
                label = "No tumor confidence"
            else:
                label = "Pituitary confidence"
            score = predictions[i]
            self.predictions_labels[i].configure(text=f"{label}: {score:.2%}")

        

        #new_pred[0] == "glioma"
        #new_pred[1] == "pituitary"


        #percent = prediction * 100

        #ImageShow.show(image)
        #self.canvas.delete('all')
        #photo = ImageTk.PhotoImage(image)
        #print(photo)
        #self.canvas.create_image(0, 0, anchor='nw', image=photo)
        #self.image_label.configure(image=photo)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        #confidence = percent[0][prediction]
        #self.first_prediction.configure(text=f'Predicted type: {first_score} (Prediction confidence: {confidence:.2f}%)')
        #self.second_prediction.configure(text=f'Predicted type: {second_score} (Prediction confidence: {confidence:.2f}%)')
        #self.third_prediction.configure(text=f'Predicted type: {third_score} (Prediction confidence: {confidence:.2f}%)')
        #self.fourth_prediction.configure(text=f'Predicted type: {fourth_score} (Prediction confidence: {confidence:.2f}%)')
        #print("first score", first_score)
        #print("sec score", second_score)
        #print("thrd score", third_score)

        #grad-cam images
        def gradCam(image, intensity=0.5):
            with tf.GradientTape() as tape:
                final_layer = self.model.get_layer('conv2d_11')
        

        

if __name__ == '__main__':
    app = App()
    app.mainloop()
