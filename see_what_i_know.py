#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 10:40:39 2017

@author: palash
"""
import os.path
import numpy as np
import tensorflow as tf
import os

import cv2

frame_height = 224
frame_width = 224
data_std = 112
data_mean = 112
def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def read_tensor_from_image_file(image, input_height=128, input_width=128,
				input_mean=0, input_std=60):
    float_caster = tf.cast(image, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label
    
def testFrames(graph,
               input_layer,
               output_layer,
               input_height=128,
               input_width=128,
               input_mean=64,
               input_std=64):
    try:
        cap = cv2.VideoCapture(0)
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)
        
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # Display the resulting framed
            t = read_tensor_from_image_file(
                    frame,input_height=input_height,
                    input_width=input_width,
                    input_mean=input_mean,
                    input_std=input_std)
            with tf.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0],
                                  {input_operation.outputs[0]: t})
            results = np.squeeze(results)
    
            top_k = results.argsort()[-5:][::-1]
            labels = load_labels(label_file)
            for i in top_k:
                print(labels[i], results[i])
            cv2.putText(frame,"Prediction: "+labels[top_k[0]]+" Confidence: "+str(results[top_k[0]])+' Press q to exit.',(25,25),cv2.FONT_HERSHEY_PLAIN,1.0,(255,0,0),thickness=2)
            cv2.imshow('Prediction Window', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    finally:
        cap.release()
        cv2.destroyAllWindows()
if(os.path.isfile("output_graph.pb") and os.path.isfile("output_graph.pb")):
    graph = load_graph("output_graph.pb")
    label_file = "output_labels.txt"
    testFrames(graph=graph,
               input_height=frame_height,
               input_width=frame_width,
               input_std=data_std,
               input_mean=data_mean,
               input_layer='input',
               output_layer='final_result')
else:
    print("You haven't taught me anythin yet. Run teach_me.py first.")



