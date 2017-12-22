"""
Copyright (c) 2017, by the Authors: Amir H. Abdi
This software is freely available under the MIT Public License. 
Please see the License file in the root for details.
"""
# uncomment the following lines to alter the default values set above
#input_fld = ''
#weight_file = 'res_net50.hdf5'
#num_output = 1
#write_graph_def_ascii_flag = True
#prefix_output_node_names_of_final_network = 'res_net50'
#output_graph_name = 'res_net_skoda.pb'

# setting input arguments
import argparse
parser = argparse.ArgumentParser(description='set input arguments')
parser.add_argument('-input_fld', action="store",
                    dest='input_fld', type=str, default='.')

parser.add_argument('-output_fld', action="store",
                    dest='output_fld', type=str, default='.')

parser.add_argument('-input_model_file', action="store",
                    dest='input_model_file', type=str, default='model.h5')

parser.add_argument('-output_model_file', action="store",
                    dest='output_model_file', type=str, default='model.pb')

parser.add_argument('-output_graphdef_file', action="store",
                    dest='output_graphdef_file', type=str, default='model.ascii')

parser.add_argument('-num_outputs', action="store",
                    dest='num_outputs', type=int, default=1)

parser.add_argument('-graph_def', action="store",
                    dest='graph_def', type=bool, default=False)

parser.add_argument('-output_node_prefix', action="store",
                    dest='output_node_prefix', type=str, default='output_node')

parser.add_argument('-f')
args = parser.parse_args()
print('input args: ', args)

from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K

output_fld = input_fld + 'tensorflow_model/'
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
weight_file_path = osp.join(input_fld, weight_file)

K.set_learning_phase(0)
net_model = load_model(weight_file_path)

pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
    pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)

sess = K.get_session()

if write_graph_def_ascii_flag:
    f = 'only_the_graph_def.pb.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
    print('saved the graph definition in ascii format at: ', osp.join(output_fld, f))

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))

