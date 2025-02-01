import tensorflow as tf
import numpy as np

# Path to the upgraded model
MODEL_DIR = "./new_saved_model"

def get_model_signature(sess):
    """Extracts input and output tensor names."""
    meta_graph = tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], MODEL_DIR)
    signature = meta_graph.signature_def

    print("\n📌 Available Signatures:", list(signature.keys()))


def load_model():
    """Loads the TensorFlow 1.15 model and returns the session and graph."""
    sess = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], MODEL_DIR)
    print("✅ Model loaded successfully.")
    return sess, sess.graph

def get_model_signature(sess):
    """Extracts input and output tensor names."""
    meta_graph = tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], MODEL_DIR)
    signature = meta_graph.signature_def

    print("\n📌 Model Signature Details:")
    for key, sig in signature.items():
        print(f"Signature: {key}")
        for input_key, input_tensor in sig.inputs.items():
            print(f"  🔹 Input: {input_key} -> {input_tensor.name}")
        for output_key, output_tensor in sig.outputs.items():
            print(f"  🔸 Output: {output_key} -> {output_tensor.name}")

    # Return first input/output tensor names found
    input_name = list(signature["serving_default"].inputs.values())[0].name
    output_name = list(signature["serving_default"].outputs.values())[0].name
    return input_name, output_name

def run_inference(sess, input_name, output_name):
    """Runs a test inference with dummy data."""
    input_tensor = sess.graph.get_tensor_by_name(input_name)
    output_tensor = sess.graph.get_tensor_by_name(output_name)

    # Adjust shape based on your model (Example: Image model with 224x224 RGB input)
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

    predictions = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
    print("\n✅ Model Inference Output:", predictions)

if __name__ == "__main__":
    sess, graph = load_model()
    # input_tensor_name, output_tensor_name = get_model_signature(sess)
    # print("input tensor name: ", input_tensor_name, "output tensor name: " , output_tensor_name)
    # run_inference(sess, input_tensor_name, output_tensor_name)
    # sess.close()
    # get_model_signature(sess)
