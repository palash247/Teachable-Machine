#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 10:41:40 2017

@author: palash
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import cv2
import os
import shutil
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import time
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

frame_height = 224
frame_width = 224
data_std = 112
data_mean = 112
IMAGES_TO_CAPTURE = 400
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1 
image_dir='./data'
output_graph='output_graph.pb'
intermediate_output_graphs_dir='./intermediate_graph/'
intermediate_store_frequency=0
output_labels='./output_labels.txt'
summaries_dir='./retrain_logs'
how_many_training_steps=200
learning_rate=0.0001
testing_percentage=20
validation_percentage=20
eval_step_interval=50
train_batch_size=32
test_batch_size=-1
validation_batch_size=-1
print_misclassified_test_images=False
model_dir='./imagenet'
bottleneck_dir='./bottleneck'
final_tensor_name='final_result'
flip_left_right=True
random_crop=0
random_scale=30
random_brightness=30
architecture='mobilenet_1.0_224' 


def classes_collector():
    dataPath = "./data/"
    ensure_dir_exists(dataPath)
    
    while(True):
        path, dirs, files = os.walk(dataPath).__next__()
        print('#######################################################')
        print('Number of classes currently recorded: ',len(dirs))
        print('Please enter a string you want me to associate to the traing data.')
        print('Or type DONE wehen you are done with the teaching stuff.')
        print('You can also make me unlearn everythin by typing RESET.')
        print('Note: I should have atleast two classes for prediction.')
        class_name = str(input())
        if(class_name.lower() == 'done'):
            break
        if(class_name.lower() == 'reset'):
            for each_class in dirs:
                shutil.rmtree(dataPath + each_class)
            continue
        capture_training_data(class_name)
        
def capture_training_data(class_name):
    try:
        cap = cv2.VideoCapture(0)
        count = 1
        dataPath = "./data/"
        ensure_dir_exists(dataPath)
        if(os.path.isdir(dataPath+class_name) == True):
            shutil.rmtree(dataPath+class_name)
        os.makedirs(dataPath+class_name)
        warning_counter = 0
        tic = time.time()
        while(count<IMAGES_TO_CAPTURE):
            ret, frame = cap.read()
            warning_counter = int(time.time() - tic)
            if(warning_counter >= 5):
                count += 1
                cv2.imwrite(dataPath+class_name+'/'+class_name+str(count)+".jpeg", frame)
                cv2.putText(frame,
                        "Number of frames captured: "+str(count),
                        (25,25),
                        cv2.FONT_HERSHEY_PLAIN,
                        2.0,
                        (255,0,0),
                        thickness=2)
            else:
                cv2.putText(frame,
                            "Recording frames in "+str(5 - warning_counter)+" seconds.",
                            (25,25),
                            cv2.FONT_HERSHEY_PLAIN,
                            2.0,
                            (0,0,255),
                            thickness=2)
            
            cv2.imshow('Recording frames for class: '+class_name,frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        for i in range (1,5):
            cv2.waitKey(1)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for i in range (1,5):
            cv2.waitKey(1)

def create_image_lists(image_dir, testing_percentage,
                       validation_percentage):
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir
                         + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.'
                    + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning('WARNING: Folder has less than 20 images, which may cause issues.'
                               )
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning('WARNING: Folder {} has more than {} images. Some images will never be selected.'.format(dir_name,
                               MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = \
                hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = int(hash_name_hashed, 16) \
                % (MAX_NUM_IMAGES_PER_CLASS + 1) * (100.0
                    / MAX_NUM_IMAGES_PER_CLASS)
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < testing_percentage \
                + validation_percentage:
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
            }
    return result

def get_image_path(image_lists,label_name,index,image_dir,category):
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, architecture):
    return get_image_path(image_lists, label_name, index,
                          bottleneck_dir, category) + '_' \
        + architecture + '.txt'


def create_model_graph(model_info, model_dir):
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(model_dir,
                                  model_info['model_file_name'])
        print ('Model path: ', model_path)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            (bottleneck_tensor, resized_input_tensor) = \
                tf.import_graph_def(graph_def, name='',
                                    return_elements=[model_info['bottleneck_tensor_name'
                                    ],
                                    model_info['resized_input_tensor_name'
                                    ]])
    return (graph, bottleneck_tensor, resized_input_tensor)


def run_bottleneck_on_image(sess,image_data,image_data_tensor,decoded_image_tensor,resized_input_tensor,bottleneck_tensor):
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def maybe_download_and_extract(data_url, model_dir):
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                             float(count * block_size)
                             / float(total_size) * 100.0))
            sys.stdout.flush()

        (filepath, _) = urllib.request.urlretrieve(data_url, filepath,
                _progress)
        print ()
        statinfo = os.stat(filepath)
        tf.logging.info('Successfully downloaded', filename,
                        statinfo.st_size, 'bytes.')
        print ('Extracting file from ', filepath)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    else:
        print ('Not extracting or downloading files, model already present in disk')


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(
    bottleneck_path,
    image_lists,
    label_name,
    index,
    image_dir,
    category,
    sess,
    jpeg_data_tensor,
    decoded_image_tensor,
    resized_input_tensor,
    bottleneck_tensor,
    ):
    """Create a single bottleneck file."""

    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess,
            image_data,
            jpeg_data_tensor,
            decoded_image_tensor,
            resized_input_tensor,
            bottleneck_tensor,
            )
    except:
        raise RuntimeError('Error during processing file %s'
                           % (image_path))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(
    sess,
    image_lists,
    label_name,
    index,
    image_dir,
    category,
    bottleneck_dir,
    jpeg_data_tensor,
    decoded_image_tensor,
    resized_input_tensor,
    bottleneck_tensor,
    architecture,
    ):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(
        image_lists,
        label_name,
        index,
        bottleneck_dir,
        category,
        architecture,
        )
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(
            bottleneck_path,
            image_lists,
            label_name,
            index,
            image_dir,
            category,
            sess,
            jpeg_data_tensor,
            decoded_image_tensor,
            resized_input_tensor,
            bottleneck_tensor,
            )
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in
                             bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(
            bottleneck_path,
            image_lists,
            label_name,
            index,
            image_dir,
            category,
            sess,
            jpeg_data_tensor,
            decoded_image_tensor,
            resized_input_tensor,
            bottleneck_tensor,
            )
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()

    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation

        bottleneck_values = [float(x) for x in
                             bottleneck_string.split(',')]
    return bottleneck_values

#!/usr/bin/python
# -*- coding: utf-8 -*-


def cache_bottlenecks(
    sess,
    image_lists,
    image_dir,
    bottleneck_dir,
    jpeg_data_tensor,
    decoded_image_tensor,
    resized_input_tensor,
    bottleneck_tensor,
    architecture,
    ):
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for (label_name, label_lists) in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for (index, unused_base_name) in enumerate(category_list):
                get_or_create_bottleneck(
                    sess,
                    image_lists,
                    label_name,
                    index,
                    image_dir,
                    category,
                    bottleneck_dir,
                    jpeg_data_tensor,
                    decoded_image_tensor,
                    resized_input_tensor,
                    bottleneck_tensor,
                    architecture,
                    )

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info(str(how_many_bottlenecks)
                                    + ' bottleneck files created.')


def get_random_cached_bottlenecks(
    sess,
    image_lists,
    how_many,
    category,
    bottleneck_dir,
    image_dir,
    jpeg_data_tensor,
    decoded_image_tensor,
    resized_input_tensor,
    bottleneck_tensor,
    architecture,
    ):

    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:

    # Retrieve a random sample of bottlenecks.

        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name,
                    image_index, image_dir, category)
            bottleneck = get_or_create_bottleneck(
                sess,
                image_lists,
                label_name,
                image_index,
                image_dir,
                category,
                bottleneck_dir,
                jpeg_data_tensor,
                decoded_image_tensor,
                resized_input_tensor,
                bottleneck_tensor,
                architecture,
                )
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:

    # Retrieve all bottlenecks.

        for (label_index, label_name) in enumerate(image_lists.keys()):
            for (image_index, image_name) in \
                enumerate(image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name,
                        image_index, image_dir, category)
                bottleneck = get_or_create_bottleneck(
                    sess,
                    image_lists,
                    label_name,
                    image_index,
                    image_dir,
                    category,
                    bottleneck_dir,
                    jpeg_data_tensor,
                    decoded_image_tensor,
                    resized_input_tensor,
                    bottleneck_tensor,
                    architecture,
                    )
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return (bottlenecks, ground_truths, filenames)


def get_random_distorted_bottlenecks(
    sess,
    image_lists,
    how_many,
    category,
    image_dir,
    input_jpeg_tensor,
    distorted_image,
    resized_input_tensor,
    bottleneck_tensor,
    ):
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name,
                                    image_index, image_dir, category)
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()

    # Note that we materialize the distorted_image_data as a numpy array before
    # sending running inference on the image. This involves 2 memory copies and
    # might be optimized in other implementations.

        distorted_image_data = sess.run(distorted_image,
                {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor,
                {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck_values)
        ground_truths.append(ground_truth)
    return (bottlenecks, ground_truths)


def should_distort_images(
    flip_left_right,
    random_crop,
    random_scale,
    random_brightness,
    ):
    return flip_left_right or random_crop != 0 or random_scale != 0 \
        or random_brightness != 0


def add_input_distortions(
    flip_left_right,
    random_crop,
    random_scale,
    random_brightness,
    input_width,
    input_height,
    input_depth,
    input_mean,
    input_std,
    ):

    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data,
            channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + random_crop / 100.0
    resize_scale = 1.0 + random_scale / 100.0
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
            minval=1.0, maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
            precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d, [input_height,
                                   input_width, input_depth])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - random_brightness / 100.0
    brightness_max = 1.0 + random_brightness / 100.0
    brightness_value = tf.random_uniform(tensor_shape.scalar(),
            minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    offset_image = tf.subtract(brightened_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
    return (jpeg_data, distort_result)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_training_ops(
    class_count,
    final_tensor_name,
    bottleneck_tensor,
    bottleneck_tensor_size,
    quantize_layer,
    learning_rate
    ):
    with tf.name_scope('input'):
        bottleneck_input = \
            tf.placeholder_with_default(bottleneck_tensor, shape=[None,
                bottleneck_tensor_size],
                name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32, [None,
                class_count], name='GroundTruthInput')
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = \
                tf.truncated_normal([bottleneck_tensor_size,
                                    class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value,
                    name='final_weights')
            if quantize_layer:
                quantized_layer_weights = \
                    quant_ops.MovingAvgQuantize(layer_weights,
                        is_training=True)
                variable_summaries(quantized_layer_weights)

            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]),
                    name='final_biases')
            if quantize_layer:
                quantized_layer_biases = \
                    quant_ops.MovingAvgQuantize(layer_biases,
                        is_training=True)
                variable_summaries(quantized_layer_biases)

            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            if quantize_layer:
                logits = tf.matmul(bottleneck_input,
                                   quantized_layer_weights) \
                    + quantized_layer_biases
                logits = quant_ops.MovingAvgQuantize(
                    logits,
                    init_min=-32.0,
                    init_max=32.0,
                    is_training=True,
                    num_bits=8,
                    narrow_range=False,
                    ema_decay=0.5,
                    )
                tf.summary.histogram('pre_activations', logits)
            else:
                logits = tf.matmul(bottleneck_input, layer_weights) \
                    + layer_biases
                tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input,
                logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = \
            tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input,
            ground_truth_input, final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction,
                    tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = \
                tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return (evaluation_step, prediction)


def save_graph_to_file(sess, graph, graph_file_name, final_tensor_name):
    output_graph_def = graph_util.convert_variables_to_constants(sess,
            graph.as_graph_def(), [final_tensor_name])

    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return


def prepare_file_system(summaries_dir,
                        intermediate_store_frequency,
                        intermediate_output_graphs_dir):
    if tf.gfile.Exists(summaries_dir):
        tf.gfile.DeleteRecursively(summaries_dir)
    tf.gfile.MakeDirs(summaries_dir)
    if intermediate_store_frequency > 0:
        ensure_dir_exists(intermediate_output_graphs_dir)
    return


def create_model_info(architecture):
    architecture = architecture.lower()
    is_quantized = False
    if architecture.startswith('mobilenet_'):
        parts = architecture.split('_')
        if len(parts) != 3 and len(parts) != 4:
            tf.logging.error("Couldn't understand architecture name '%s'"
                             , architecture)
            return None
        version_string = parts[1]
        if version_string != '1.0' and version_string != '0.75' \
            and version_string != '0.50' and version_string != '0.25':
            tf.logging.error(""""The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
  but found '%s' for architecture '%s'"""
                             , version_string, architecture)
            return None
        size_string = parts[2]
        if size_string != '224' and size_string != '192' \
            and size_string != '160' and size_string != '128':
            tf.logging.error("""The Mobilenet input size should be '224', '192', '160', or '128',
 but found '%s' for architecture '%s'"""
                             , size_string, architecture)
            return None
        if len(parts) == 3:
            is_quantized = False
        else:
            if parts[3] != 'quantized':
                tf.logging.error("Couldn't understand architecture suffix '%s' for '%s'"
                                 , parts[3], architecture)
                return None
            is_quantized = True

        if is_quantized:
            data_url = \
                'http://download.tensorflow.org/models/mobilenet_v1_'
            data_url += version_string + '_' + size_string \
                + '_quantized_frozen.tgz'
            bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
            resized_input_tensor_name = 'Placeholder:0'
            model_dir_name = 'mobilenet_v1_' + version_string + '_' \
                + size_string + '_quantized_frozen'
            model_base_name = 'quantized_frozen_graph.pb'
        else:

            data_url = \
                'http://download.tensorflow.org/models/mobilenet_v1_'
            data_url += version_string + '_' + size_string \
                + '_frozen.tgz'
            bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
            resized_input_tensor_name = 'input:0'
            model_dir_name = 'mobilenet_v1_' + version_string + '_' \
                + size_string
            model_base_name = 'frozen_graph.pb'

        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        model_file_name = os.path.join(model_dir_name, model_base_name)
        input_mean = 127.5
        input_std = 127.5
    else:
        tf.logging.error("Couldn't understand architecture name '%s'",
                         architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
        'quantize_layer': is_quantized,
        }


def add_jpeg_decoding(
    input_width,
    input_height,
    input_depth,
    input_mean,
    input_std,
    ):
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data,
            channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
            resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return (jpeg_data, mul_image)


def main():
    
    
    tf.logging.set_verbosity(tf.logging.INFO)

  # Prepare necessary directories that can be used during training

    prepare_file_system(summaries_dir,
                        intermediate_store_frequency,
                        intermediate_output_graphs_dir)

  # Gather information about the model architecture we'll be using.

    model_info = create_model_info(architecture)
    if not model_info:
        tf.logging.error('Did not recognize architecture flag')
        return -1

  # Set up the pre-trained graph.

    maybe_download_and_extract(model_info['data_url'],model_dir)
    (graph, bottleneck_tensor, resized_image_tensor) = \
        create_model_graph(model_info, model_dir)

  # Look at the folder structure, and create lists of all the images.

    image_lists = create_image_lists(image_dir,
                                     testing_percentage,
                                     validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at '
                         + image_dir)
        return -1
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at '
                         + image_dir
                         + ' - multiple classes are needed for classification.'
                         )
        return -1

  # See if we're applying any distortion on images.

    do_distort_images = should_distort_images(flip_left_right,
                                              random_crop,
                                              random_scale,
                                              random_brightness)

    with tf.Session(graph=graph) as sess:

    # Set up the image decoding sub-graph.

        (jpeg_data_tensor, decoded_image_tensor) = \
            add_jpeg_decoding(model_info['input_width'],
                              model_info['input_height'],
                              model_info['input_depth'],
                              model_info['input_mean'],
                              model_info['input_std'])

        if do_distort_images:

      # We will be applying distortions, so setup the operations we'll need.

            (distorted_jpeg_data_tensor, distorted_image_tensor) = \
                add_input_distortions(
                        flip_left_right,
                        random_crop,
                        random_scale,
                        random_brightness,
                model_info['input_width'],
                model_info['input_height'],
                model_info['input_depth'],
                model_info['input_mean'],
                model_info['input_std'],
                )
        else:

      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.

            cache_bottlenecks(
                sess,
                image_lists,
                image_dir,
                bottleneck_dir,
                jpeg_data_tensor,
                decoded_image_tensor,
                resized_image_tensor,
                bottleneck_tensor,
                architecture,
                )

    # Add the new layer that we'll be training.

        (train_step, cross_entropy, bottleneck_input,
         ground_truth_input, final_tensor) = \
            add_final_training_ops(len(image_lists.keys()),
                                   final_tensor_name,
                                   bottleneck_tensor,
                                   model_info['bottleneck_tensor_size'
                                   ], model_info['quantize_layer'],
                                              learning_rate)

    # Create the operations we need to evaluate the accuracy of our new layer.

        (evaluation_step, prediction) = \
            add_evaluation_step(final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to the summaries_dir

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir
                + '/train', sess.graph)

        validation_writer = tf.summary.FileWriter(summaries_dir
                + '/validation')

    # Set up all our weights to their initial default values.

        init = tf.global_variables_initializer()
        sess.run(init)

    # Run the training for as many cycles as requested on the command line.

        for i in range(how_many_training_steps):

      # Get a batch of input bottleneck values, either calculated fresh every
      # time with distortions applied, or from the cache stored on disk.

            if do_distort_images:

                (train_bottlenecks, train_ground_truth) = \
                    get_random_distorted_bottlenecks(
                    sess,
                    image_lists,
                    train_batch_size,
                    'training',
                    image_dir,
                    distorted_jpeg_data_tensor,
                    distorted_image_tensor,
                    resized_image_tensor,
                    bottleneck_tensor,
                    )
            else:

                (train_bottlenecks, train_ground_truth, _) = \
                    get_random_cached_bottlenecks(
                    sess,
                    image_lists,
                    train_batch_size,
                    'training',
                    bottleneck_dir,
                    image_dir,
                    jpeg_data_tensor,
                    decoded_image_tensor,
                    resized_image_tensor,
                    bottleneck_tensor,
                    architecture,
                    )

      # Feed the bottlenecks and ground truth into the graph, and run a training
      # step. Capture training summaries for TensorBoard with the `merged` op.

            (train_summary, _) = sess.run([merged, train_step],
                    feed_dict={bottleneck_input: train_bottlenecks,
                    ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

      # Every so often, print out how well the graph is training.

            is_last_step = i + 1 == how_many_training_steps
            if i % eval_step_interval == 0 or is_last_step:
                (train_accuracy, cross_entropy_value) = \
                    sess.run([evaluation_step, cross_entropy],
                             feed_dict={bottleneck_input: train_bottlenecks,
                             ground_truth_input: train_ground_truth})
                tf.logging.info('%s: Step %d: Train accuracy = %.1f%%'
                                % (datetime.now(), i, train_accuracy
                                * 100))
                tf.logging.info('%s: Step %d: Cross entropy = %f'
                                % (datetime.now(), i,
                                cross_entropy_value))
                (validation_bottlenecks, validation_ground_truth, _) = \
                    get_random_cached_bottlenecks(
                    sess,
                    image_lists,
                    validation_batch_size,
                    'validation',
                    bottleneck_dir,
                    image_dir,
                    jpeg_data_tensor,
                    decoded_image_tensor,
                    resized_image_tensor,
                    bottleneck_tensor,
                    architecture,
                    )

        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.

                (validation_summary, validation_accuracy) = \
                    sess.run([merged, evaluation_step],
                             feed_dict={bottleneck_input: validation_bottlenecks,
                             ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)'
                                 % (datetime.now(), i,
                                validation_accuracy * 100,
                                len(validation_bottlenecks)))

      # Store intermediate results

            intermediate_frequency = intermediate_store_frequency

            if intermediate_frequency > 0 and i \
                % intermediate_frequency == 0 and i > 0:
                intermediate_file_name = \
                    intermediate_output_graphs_dir \
                    + 'intermediate_' + str(i) + '.pb'
                tf.logging.info('Save intermediate result to : '
                                + intermediate_file_name)
                save_graph_to_file(sess, graph, intermediate_file_name, final_tensor_name)

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.

        (test_bottlenecks, test_ground_truth, test_filenames) = \
            get_random_cached_bottlenecks(
            sess,
            image_lists,
            test_batch_size,
            'testing',
            bottleneck_dir,
            image_dir,
            jpeg_data_tensor,
            decoded_image_tensor,
            resized_image_tensor,
            bottleneck_tensor,
            architecture,
            )
        (test_accuracy, predictions) = sess.run([evaluation_step,
                prediction],
                feed_dict={bottleneck_input: test_bottlenecks,
                ground_truth_input: test_ground_truth})
        tf.logging.info('Final test accuracy = %.1f%% (N=%d)'
                        % (test_accuracy * 100, len(test_bottlenecks)))

        if print_misclassified_test_images:
            tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
            for (i, test_filename) in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    tf.logging.info('%70s  %s' % (test_filename,
                                    list(image_lists.keys())[predictions[i]]))

    # Write out the trained graph and labels with the weights stored as
    # constants.

        save_graph_to_file(sess, graph, output_graph,final_tensor_name)
        with gfile.FastGFile(output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')
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
    
def test_frames(graph,
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
            cv2.putText(frame,"Prediction: "+labels[top_k[0]]+" Confidence: "+str(results[top_k[0]])+'\n Press q to exit.',(25,25),cv2.FONT_HERSHEY_PLAIN,2.0,(255,0,0),thickness=2)
            cv2.imshow('Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    finally:
        cap.release()
        cv2.destroyAllWindows()
classes_collector()
main()
graph = load_graph("output_graph.pb")
label_file = "output_labels.txt"
test_frames(graph=graph,
           input_height=frame_height,
           input_width=frame_width,
           input_std=data_std,
           input_mean=data_mean,
           input_layer='input',
           output_layer='final_result')
