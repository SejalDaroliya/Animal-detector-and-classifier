import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from PIL import Image, ImageTk
import numpy as np

# Load the trained model
model = load_model('animal_classifier_with_diet.h5')

# Define class names and diet names
classes = ['butterfly', 'cat', 'cow', 'dog', 'elephant', 'hen', 'horse', 'sheep', 'spider', 'squirrel']
diets = ['herbivore', 'carnivore', 'herbivore', 'carnivore', 'herbivore', 'herbivore', 'herbivore', 'herbivore', 'carnivore', 'herbivore']


# Create a simple GUI for predictions
def load_image(filename):
    img = Image.open(filename)
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image():
    filepath = filedialog.askopenfilename()
    if filepath:
        img = load_image(filepath)
        class_pred, diet_pred = model.predict(img)
        predicted_class = classes[np.argmax(class_pred)]
        predicted_diet = diets[np.argmax(diet_pred)]
        result_label.config(text=f"Predicted: {predicted_class} ({predicted_diet})")
        img = Image.open(filepath)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

# Create the main window
root = tk.Tk()
root.title("Animal Classifier with Diet Prediction")

panel = tk.Label(root)
panel.pack(side="top", fill="both", expand="yes")

btn = tk.Button(root, text="Load Image", command=predict_image)
btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

result_label = tk.Label(root, text="Prediction will appear here")
result_label.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

root.mainloop()
