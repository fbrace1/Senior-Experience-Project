import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load('C:/Users/FBRAC/Projects/DataSavedFromProgramRun/data.npy', allow_pickle=True)

# Function to display a batch of images
def display_batch(data, batch_size=16, start_index=0):
    end_index = min(start_index + batch_size, len(data))
    num_images = end_index - start_index
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        image = data[start_index + i][0].reshape(180, 180)
        label = data[start_index + i][1]
        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')
    plt.show()

# Display images in batches
batch_size = 16
for start_index in range(0, len(data), batch_size):
    display_batch(data, batch_size=batch_size, start_index=start_index)
    input("Press Enter to continue...")

