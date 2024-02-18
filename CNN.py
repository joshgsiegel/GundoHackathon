#This is where the convolutional neural network is defined and trained
# we will also be doing transfer learning here on a natural language data set
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Construct the head of the model that will be placed on top of the base model
model = Sequential()
model.add(baseModel)
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))  # Change this to match your number of classes

# Loop over all layers in the base model and freeze them so they will not be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

# Compile our model (this needs to be done after our setting our layers to non-trainable)
print("[INFO] compiling model...")
opt = Adam(lr=1e-4, decay=1e-4 / 20)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])