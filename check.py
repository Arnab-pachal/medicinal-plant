import tensorflow as tf
import cv2

# Preprocess the test image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Resize the image to match the input size of the model
    image = cv2.resize(image, (200, 200))
    # Rescale pixel values to the range [0, 1]
    image = image / 255.0
    return image

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Path to the test image
test_image_path = r'C:\Users\ASUS\OneDrive\Desktop\dekstop\MY PROJECT\medicinal plant\training\Amla\31.jpg'

# Preprocess the test image
test_image = preprocess_image(test_image_path)

# Expand dimensions to match the input shape expected by the model
test_image = tf.expand_dims(test_image, axis=0)


# Predict the class probabilities
predictions = model.predict(test_image)

# Post-process the prediction
predicted_class = tf.argmax(predictions, axis=1).numpy()[0]



# Visualize the result
print(f'Predicted Class: {predicted_class}')
