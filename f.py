import tensorflow as tf

with tf.Session(graph=tf.Graph()) as sess: 
	tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], export_dir) 
	export_dir = "./new_saved_model" # Get signature definition
    	meta_graph = tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], export_dir) 
	signature = meta_graph.signature_def 
with tf.Session(graph=tf.Graph()) as sess: # Extract input and output tensor names
    tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], export_dir) for key, sig in signature.items(): print(f"Signature: {key}") for input_key, input_tensor in 
		sig.inputs.items():
    print("Model loaded successfully.") print(f"Input: {input_key} -> {input_tensor.name}") for output_key, output_tensor in sig.outputs.items(): print(f"Output: 
            		{output_key} -> {output_tensor.name}")
    # List operations to check integrity
    for op in sess.graph.get_operations(): print(op.name)
