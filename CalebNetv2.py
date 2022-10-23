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
subs_src = []
target = []

# Import thumbnails from dataset/ folder
for img in os.listdir('dataset'):
	source.append(loadImage(f'dataset/{img}'))
	views, subs = img[12:-4].split("_")
	views_mag = len(views) - 1
	target.append(views_mag)
	subs_mag = (len(views) - 1) / 10 # some dog shit normalisation
	subs_src.append(subs_mag)

# Convert to numpy arrays
source = np.array(source)
target = np.array(target)
subs_src = np.array(subs_src)
len(source)

# %%
# x_train, x_valid, y_train, y_valid = train_test_split(source, subs_src, target, test_size=0.1)
# needed a train_test_split that does two things
train_to = int(len(source) / 10 * 8)
valid_from = train_to
valid_to = len(source)

x_train_p1 = np.array(source[0: train_to])
x_train_p2 = np.array(subs_src[0: train_to])
y_train = np.array(target[0: train_to])

x_train = [x_train_p1, x_train_p2]

x_valid_p1 = np.array(source[valid_from: valid_to])
x_valid_p2 = np.array(subs_src[valid_from: valid_to])
y_valid = np.array(target[valid_from: valid_to])

x_valid = [x_valid_p1, x_valid_p2]

# %%
i1 = keras.Input(shape=(img_size[1], img_size[0], 3), name="thumbnail")
x = keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='Same', activation='relu', name="feat_ext_1")(i1)
x = keras.layers.MaxPooling2D(pool_size=2, strides=2, name="pool_1")(x)
x = keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='Same', activation='relu', name="feat_ext_2")(x)
x = keras.layers.MaxPooling2D(pool_size=2, strides=2, name="pool_2")(x)
x = keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='Same', activation='relu', name="feat_ext_3")(x)
x = keras.layers.MaxPooling2D(pool_size=2, strides=2, name="pool_3")(x)
x = keras.layers.Flatten(name="flatten_1")(x)
x = keras.layers.Dropout(0.5, name="dropout_0.5")(x)
x = keras.layers.Dense(512, activation='relu', name="thumb_dense")(x)

i2 = keras.Input(shape=(1,), name="subs")
y = keras.layers.Dense(512, activation='relu', name="sub_expansion")(i2)

z = keras.layers.Concatenate(name="concat")([x, y])
z = keras.layers.Dense(512, activation='relu', name="concat_dense")(z)

output = keras.layers.Dense(10, activation='softmax', name="output")(z)
model = keras.Model(inputs=[i1, i2], outputs=output)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=keras.losses.SparseCategoricalCrossentropy())
model.summary()

# %%
batch_size = 64
epochs = 64

history = model.fit(
	x_train, y_train,
	batch_size=batch_size,
	epochs=epochs,
	validation_data=(x_valid, y_valid)#,
	#steps_per_epoch=len(x_train) // batch_size # num_samples / batch_size
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
num_results = 32#len(preds)
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

# %%
img = loadImage("maxresdefault.jpg")
thing1 = np.array([img])
thing2 = np.array([9])
np.argmax(model.predict([thing1, thing2]), axis=1)


