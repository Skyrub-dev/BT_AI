from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import time
import numpy as np
import tensorflow as tf
from keras import backend as K
import cv2

root = Tk()
root.wm_title("Tumor classification")
root.maxsize(1920, 1080)
root.resizable(width = True, height = True)

model = tf.keras.models.load_model('C:/Users/tomke/Downloads/model/bt_ai_EXP3-91acc-04lss-08-04-23.h5', compile=False)

canvas = Canvas(root, width=1920, height=1080)
canvas.grid(row=0, column=1, padx=10, pady=5)

# Create left and right frames

top_frame = Frame(canvas, width=1920, height=50, bg='grey')
canvas.create_window(0, 0, anchor='nw', window=top_frame, width=1920)



Button(top_frame, text='Export as PNG/JPG').pack(side=RIGHT, padx=5, pady=5)
Button(top_frame, text='Export as DCIOM').pack(side=RIGHT, padx=5, pady=5)

left_frame = Frame(canvas, width=200, height=400, bg='grey')
canvas.create_window(0, 50, anchor='nw', window=left_frame)

augment_title = Label(left_frame, text=f"Augmentation settings", bg="grey", fg="black")
augment_title.grid(row=1, column=0, padx=5, pady=5, sticky="NE")
#Button(left_frame, text='Button 3').pack(side=LEFT, padx=5, pady=5)
#Button(left_frame, text='Button 4').pack(side=LEFT, padx=5, pady=5)
#Button(left_frame, text='Button 5').pack(side=LEFT, padx=5, pady=5)


right_frame = Canvas(root, width=650, height=400, bg='black')
canvas.create_window(250, 50, anchor='nw', window=right_frame)

#labels on image


# Create frames and labels in left_frame
#Label(left_frame, text="Original Image").grid(row=0, column=0, padx=5, pady=5)



def info():
    window = Toplevel()
    window.wm_title("Info")
    # Add widgets to the window
    label = Label(window, text="CNN model statistics:\n93% acc\n View from: github.com/fgh")
    label.pack()

#def gradCam(image, intensity=0.5):
#    with tf.GradientTape() as tape:
#        final_layer = model.get_layer('conv2d_11')
#        iterate = tf.keras.models.Model([model.inputs], [model.output, final_layer.output])
#        model_out, final_layer = iterate(x)
#        class_out = model_out[:, np.argmax(model_out[0])]
#        grads = tape.gradient(class_out, final_layer)
#        pooled_grads = K.mean(grads, axis=(0, 1, 2))
#    
#    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, final_layer), axis=-1)
#    heatmap = np.maximum(heatmap, 0)
#    heatmap /= np.max(heatmap)
#    heatmap = heatmap.reshape((8, 8))
#
#    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

def open_img():
    file = filedialog.askopenfilename()
    image = Image.open(file)
    photo = ImageTk.PhotoImage(image)
    
    #inits for ml model
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    
    img_label = Label(right_frame, width=1280, height=720, bg='black')
    img_label.image = photo  # assign the photo to the label's instance variable
    img_label.config(image=photo)
    img_label.grid(row=1, column=0, padx=5, pady=5)
    
    #Filename
    filename = file.split("/")[-1]
    name_label = Label(right_frame, text=f"{filename}", bg="black", fg="white")
    name_label.grid(row=0, column=0, padx=5, pady=5, sticky="NE")

    #height and width
    heightweight_label = Label(right_frame, text=f"VIEW SIZE: (650, 400)\nIMG SIZE:{image.size}", bg="black", fg="white")
    heightweight_label.grid(row=1, column=0, padx=5, pady=5, sticky="NE")

    #Classification types - REPLACE THIS WHEN THE MODEL IS INTEGRATED
    classification_label = []
    for i in range(4):
        class_label = Label(right_frame, text=f"Model precicto", bg="black", fg="White")
        classification_label.append(class_label)
        class_label.grid(row=i+1, column=0, padx=2, pady=2, sticky="SW")

    predictions = model.predict(image_array)[0]
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
        classification_label[i].configure(text=f"{label}: {score:.2%}")

    #misc
    zoom_label = Label(right_frame, text=f"Zoom: 100%", bg="black", fg="White")
    zoom_label.grid(row=1, column=0, padx=5, pady=5, sticky="SE")
    
    editor_label = Label(right_frame, text=f"Created with BT_AI", bg="black", fg="white")
    editor_label.grid(row=0, column=0, padx=5, pady=5, sticky="NW")
    
    date_time_label = Label(right_frame, text=f"{time.strftime('%m/%d/%Y %H:%M:%S')}", bg="black", fg="white")
    date_time_label.grid(row=1, column=0, padx=5, pady=5, sticky="NW")

    with tf.GradientTape() as tape:
        final_layer = model.get_layer('conv2d_11')
        iterate = tf.keras.models.Model([model.inputs], [model.output, final_layer.output])
        model_out, final_layer = iterate(image_array)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, final_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, final_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((8, 8))

last_x = None
last_y = None

def paint(event):
    global last_x, last_y
    if last_x and last_y:
        x1, y1 = last_x, last_y
    else:
        x1, y1 = event.x, event.y
    x2, y2 = event.x, event.y
    last_x, last_y = x2, y2
    python_green = "#FFFFFF"
    right_frame.create_line(x1, y1, x2, y2, fill=python_green, width=2)
    
def reset(event):
    global last_x, last_y
    last_x, last_y = None, None
    
right_frame.bind("<ButtonRelease-1>", reset)
right_frame.bind("<B1-Motion>", paint)

#def gradcam(image, intensity=0.5)

Button(top_frame, text='open image', command = open_img).pack(side=LEFT, padx=5, pady=5)
Button(top_frame, text='About model', command=info).pack(side=LEFT, padx=5, pady=5)
#Button(left_frame, text='Apply heatmap').pack(side=LEFT, padx=5, pady=5)

#Initial warning message
messagebox.showinfo("Information", "Before using this software, please be aware that any AI predictions may not be fully accurate. For more information, visit the Github page, or click 'About model' for more info")
root.mainloop()