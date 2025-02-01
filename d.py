import tensorflow as tf
# Check TensorFlow version
print("TensorFlow version:", tf.__version__)
# Load the SavedModel
export_dir = "./saved_model" # Adjust this to your actual model directory 
with tf.Session(graph=tf.Graph()) as sess: 
    tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], export_dir)
    print("Model loaded successfully.")
    # Export the model again using TensorFlow 1.15
    builder = tf.saved_model.builder.SavedModelBuilder("./new_saved_model") 
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.SERVING]) 
    builder.save()
print("Model successfully upgraded to TensorFlow 1.15.")
