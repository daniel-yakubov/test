import tensorflow as tf
# Load the saved model from the .pb file
model = tf.saved_model.load("./saved_model")

model.save("new_mdoel")
