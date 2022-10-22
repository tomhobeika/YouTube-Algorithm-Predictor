from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def loadImage(path):
	# Read image and resize to 480p
	img = Image.open(path, 'r').convert('RGB').resize((640, 480))
	# Convert colors from 0-255 to 0-1
	return np.asarray(img) / 255

# AI will try to match target output given source as input
source = []
target = []
ids = []

# Import thumbs from /dataset/ folder
for img in os.listdir('dataset'):
	source.append(loadImage(f'dataset/{img}'))
	ids.append(img[:11])
	views = int(img[12:].split("_")[0])
	target.append(views)

# Convert stuff to numpy arrays
source = np.array(source)
target = np.array(target)

# Create basic sequential model
model = keras.models.Sequential(
	[
		# Input is 640 x 480 x 3 color channels (RGB)
		keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='Same', activation='relu', input_shape=(480, 640, 3)),
		keras.layers.MaxPooling2D(pool_size=2),

		keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='Same', activation='relu'),
		keras.layers.MaxPooling2D(pool_size=2, strides=2),

		keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, padding='Same', activation='relu'),
		keras.layers.MaxPooling2D(pool_size=2, strides=2),

		keras.layers.Flatten(),
		keras.layers.Dense(16384),
		keras.layers.Dense(4096),
		keras.layers.Dense(1)
		# Output is 1D view count
	]
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mae')
model.summary()

batch_size = 64
epochs = 128

model.fit(source, target,
	batch_size=batch_size,
	epochs=epochs,
	steps_per_epoch=source.shape[0] // batch_size # num_samples / batch_size
)

preds = model.predict(source).flatten()

# Plot results
num_results = 10
width = 0.35
fig, ax = plt.subplots()
x = np.arange(num_results)
rects1 = ax.bar(x - width / 2, preds[:num_results], width, label='Predicted')
rects2 = ax.bar(x + width / 2, target[:num_results], width, label='Actual')
ax.legend()
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
fig.tight_layout()
plt.show()