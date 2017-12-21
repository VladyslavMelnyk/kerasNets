import tensorflow as tf
import cv2
import numpy as np


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

# We use our "load_graph" function
graph = load_graph("./tensorflow_model/dense_net_169.pb")

# We can verify that we can access the list of operations in the graph
for op in graph.get_operations():
    print(op.name)     # <--- printing the operations snapshot below
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions

# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/input_1:0')
y = graph.get_tensor_by_name('prefix/dense_net0:0')

# We launch a Session
with tf.Session(graph=graph) as sess:
    img = "/media/lv_user/Data1/Dropbox/Data_new/skoda/test_1/JPEGImages/raw_2017-09-21_16_35_01.298416.jpg"
    im = cv2.imread(img)
    im_resized = cv2.resize(im, dsize=(600, 1000),
		                        interpolation=cv2.INTER_LINEAR)
    im_resized = np.expand_dims(im_resized, axis=0)
    test_features = im_resized
        # compute the predicted output for test_x
    pred_y = sess.run( y, feed_dict={x: test_features} )
    print(pred_y)
