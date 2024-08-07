import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import warnings
from PIL import Image

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

# Path to the Animals-10 dataset
data_dir = r"ANIMAL CLASSIFIER PROJECT 5/raw-img/"

# Define the classes and their corresponding diets
classes = ['butterfly', 'cat', 'cow', 'dog', 'elephant', 'hen', 'horse', 'sheep', 'spider', 'squirrel']
diets = ['herbivore', 'carnivore', 'herbivore', 'carnivore', 'herbivore', 'herbivore', 'herbivore', 'herbivore', 'carnivore', 'herbivore']

# Map classes to diet labels
diet_labels = {class_name: diet for class_name, diet in zip(classes, diets)}

# Helper function to load and preprocess images
def load_image(img_path):
    try:
        image = Image.open(img_path).convert('RGB')
        image = image.resize((128, 128))
        image = np.array(image) / 255.0
        return image
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

# Load the images and labels
images = []
class_labels = []
diet_labels_encoded = []

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = load_image(img_path)
        if img is not None:
            images.append(img)
            class_labels.append(classes.index(class_name))
            diet_labels_encoded.append(0 if diet_labels[class_name] == 'herbivore' else 1)
            print(f"Loaded image {img_path} from class {class_name}")

images = np.array(images)
class_labels = to_categorical(np.array(class_labels), num_classes=len(classes))
diet_labels_encoded = to_categorical(np.array(diet_labels_encoded), num_classes=2)

# Split the dataset into training and validation sets
X_train, X_val, y_class_train, y_class_val, y_diet_train, y_diet_val = train_test_split(
    images, class_labels, diet_labels_encoded, test_size=0.2, random_state=42)

print(f"Number of training samples: {len(X_train)}")
print(f"Number of validation samples: {len(X_val)}")


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Input layer
inputs = Input(shape=(128, 128, 3))

# Convolutional layers
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layers
class_output = Dense(len(classes), activation='softmax', name='class_output')(x)
diet_output = Dense(2, activation='softmax', name='diet_output')(x)

# Build and compile the model
model = Model(inputs=inputs, outputs=[class_output, diet_output])
model.compile(optimizer='adam', loss={'class_output': 'categorical_crossentropy', 'diet_output': 'categorical_crossentropy'}, metrics=['accuracy','mae'])

# Train the model
history = model.fit(X_train, {'class_output': y_class_train, 'diet_output': y_diet_train}, epochs=10, validation_data=(X_val, {'class_output': y_class_val, 'diet_output': y_diet_val}), batch_size=32)

# Save the model
model.save('animal_classifier_with_diet.h5')
