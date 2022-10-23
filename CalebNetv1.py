# %%
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os
import math

# %%
# Downscale thumbnails to 360p for training
img_size = (168, 94)

def loadImage(path):
	img = Image.open(path, 'r').convert('RGB').resize(img_size)
	# Convert colors from 0-255 to 0-1
	return np.asarray(img) / 255

# %%
# AI tries to match target values given source values
source = []
target = []

# Import thumbnails from dataset/ folder
for img in os.listdir('dataset'):
	source.append(loadImage(f'dataset/{img}'))
	views, subs = img[12:-4].split("_")
	views_mag = len(views) - 1
	target.append(views_mag)
	
# Convert to numpy arrays
source = np.array(source)
target = np.array(target)
len(source)

# %%
x_train, x_valid, y_train, y_valid = train_test_split(source, target, test_size=0.1)

# %%
model = keras.models.Sequential(
	[
		keras.Input(shape=(img_size[1], img_size[0], 3)),

		keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='Same', activation='relu'),
		keras.layers.MaxPooling2D(pool_size=2, strides=3),

		keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='Same', activation='relu'),
		keras.layers.MaxPooling2D(pool_size=2, strides=2),

		keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='Same', activation='relu'),
		keras.layers.MaxPooling2D(pool_size=2, strides=2),

		keras.layers.Flatten(),

        keras.layers.Dropout(0.5),

		keras.layers.Dense(512, activation="relu"), # idk whether to activate this

		keras.layers.Dense(10, activation="softmax")
	]
)

# 'mae' seems slightly better than 'mse'
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss='sparse_categorical_crossentropy')
model.summary()

# %%
batch_size = 16
epochs = 32

history = model.fit(
	x_train, y_train,
	batch_size=batch_size,
	epochs=epochs,
	validation_data=(x_valid, y_valid),
	steps_per_epoch=len(x_train) // batch_size # num_samples / batch_size
)

# %%
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["train", "validation"])
plt.show()

# %%
preds = model.predict(x_valid)
preds = np.argmax(preds, axis=1)
num_results = 16#len(preds)
width = 0.35
fig, ax = plt.subplots()
x = np.arange(num_results)
rects1 = ax.bar(x - width / 2, y_valid[:num_results], width, label='Actual')
rects2 = ax.bar(x + width / 2, preds[:num_results], width, label='Predicted')
ax.legend()
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
fig.tight_layout()
plt.title("Actual vs predicted views")
plt.show()


