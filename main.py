
#Importing Necessary libraries
import numpy as np
import cv2

# Paths to the model files and image
prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
kernel_path = "models/pts_in_hull.npy"
image_path = "flower.jpg"

# Load the network model from the Caffe prototxt and weights
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load the kernel points used for quantization
kernel_points = np.load(kernel_path)

# Transpose and reshape the kernel points
kernel_points = kernel_points.transpose().reshape(2, 313, 1, 1)

# Set the kernel points as blobs for the corresponding layers

class8_ab = net.getLayerId("class8_ab")
conv8_313_rh = net.getLayerId("conv8_313_rh")
net.getLayer(class8_ab).blobs = [kernel_points.astype("float32")]
net.getLayer(conv8_313_rh).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the input image
image = cv2.imread(image_path)
scaled_image = image.astype("float32") / 255.0
lab_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2LAB)

# Resize the LAB image to the required input size
resized_image = cv2.resize(lab_image, (224, 224))
L_channel = cv2.split(resized_image)[0]
L_channel -= 50

print("Colorizing the image")

# Set the L channel as input to the network
net.setInput(cv2.dnn.blobFromImage(L_channel))

# Forward pass through the network to get the ab channels
ab_channels = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize the ab channels to match the original image size
ab_channels = cv2.resize(ab_channels, (image.shape[1], image.shape[0]))

# Extract the L channel from the LAB image
L_channel = cv2.split(lab_image)[0]


# Concatenate the L and ab channels
colorized_image = np.concatenate((L_channel[:, :, np.newaxis], ab_channels), axis=2)


# Convert the LAB image to BGR color space
colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_LAB2BGR)


# Clip the pixel values between 0 and 1
colorized_image = np.clip(colorized_image, 0, 1)


# Convert the colorized image to 8-bit unsigned integer format
colorized_image = (255 * colorized_image).astype("uint8")


# Display the original and colorized images
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized_image)
cv2.waitKey(0)