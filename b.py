import tensorflow as tf
# For TensorFlow 1.15, use the v1 compatibility mode
tf.compat.v1.disable_eager_execution()
# Step 1: Load the saved model from the "saved_model" folder
with tf.compat.v1.Session() as sess:
    # Load the model
    model = tf.compat.v1.saved_model.loader.load(sess, ["serve"], "./saved_model")
    print("Model loaded successfully!")
    # Step 2: Save the model again in the TensorFlow 1.15 format
    export_dir = './updated_saved_model'
    # Ensure you adjust input and output names according to your model
    tf.compat.v1.saved_model.simple_save(sess, export_dir, inputs={"input": model.inputs}, # Adjust this to match your model's input signature 
    outputs={"output":  model.outputs} # Adjust this to match your model's output signature
    ) 
    print(f"Model saved successfully at {export_dir}")
# Step 3: (Optional) Test the updated saved model
with tf.compat.v1.Session() as sess: 
    model = tf.compat.v1.saved_model.loader.load(sess, ["serve"], "./updated_saved_model")
    print("Updated model loaded successfully!")
