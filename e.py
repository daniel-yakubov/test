import tensorflow as tf 
export_dir = "./new_saved_model" 
with tf.Session(graph=tf.Graph()) as sess: 
    tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], export_dir) 
    print("Model loaded successfully.")
    # List operations to check integrity
    for op in sess.graph.get_operations(): 
       print(op.name)
