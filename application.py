from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import time

root = Tk()
root.wm_title("Tumor classification")
root.maxsize(1920, 1080)
root.resizable(width = True, height = True)


canvas = Canvas(root, width=1920, height=1080)
canvas.grid(row=0, column=1, padx=10, pady=5)

# Create left and right frames

top_frame = Frame(canvas, width=1920, height=50, bg='grey')
canvas.create_window(0, 0, anchor='nw', window=top_frame, width=1920)


Button(top_frame, text='Button 2').pack(side=LEFT, padx=5, pady=5)
Button(top_frame, text='Export as PNG/JPG').pack(side=RIGHT, padx=5, pady=5)
Button(top_frame, text='Export as DCIOM').pack(side=RIGHT, padx=5, pady=5)

left_frame = Frame(canvas, width=200, height=400, bg='grey')
canvas.create_window(0, 50, anchor='nw', window=left_frame)

augment_title = Label(left_frame, text=f"Augmentation settings", bg="grey", fg="black")
augment_title.grid(row=1, column=0, padx=5, pady=5, sticky="NE")
#Button(left_frame, text='Button 3').pack(side=LEFT, padx=5, pady=5)
#Button(left_frame, text='Button 4').pack(side=LEFT, padx=5, pady=5)
#Button(left_frame, text='Button 5').pack(side=LEFT, padx=5, pady=5)


right_frame = Frame(canvas, width=650, height=400, bg='black')
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

def open_img():
    file = filedialog.askopenfilename()
    image = Image.open(file)
    photo = ImageTk.PhotoImage(image)
    #image = image.resize((224, 224))
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
    classification_label = Label(right_frame, text=f"Model predictions:\n Glioma: 0.2%\nMeningioma: 0.0%\nNone: 0.1%\nPituitary: 99.87%", bg="black", fg="white")
    classification_label.grid(row=1, column=0, padx=5, pady=5, sticky="SW")
    
    #misc
    zoom_label = Label(right_frame, text=f"Zoom: 100%", bg="black", fg="White")
    zoom_label.grid(row=2, column=0, padx=5, pady=5, sticky="SW")
    
    editor_label = Label(right_frame, text=f"Created with BT_AI", bg="black", fg="white")
    editor_label.grid(row=0, column=0, padx=5, pady=5, sticky="NW")
    
    date_time_label = Label(right_frame, text=f"{time.strftime('%m/%d/%Y %H:%M:%S')}", bg="black", fg="white")
    date_time_label.grid(row=1, column=0, padx=5, pady=5, sticky="NW")


Button(top_frame, text='open image', command = open_img).pack(side=LEFT, padx=5, pady=5)
Button(top_frame, text='About model', command=info).pack(side=LEFT, padx=5, pady=5)

root.mainloop()