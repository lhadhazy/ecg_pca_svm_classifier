from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
# sys.path.append("..")
import numpy as np
import pandas as pd
from scipy import signal
import peakutils
import tensorflow as tf
from mesonbuild.mlog import initialize
from tensorflow.python.ops.variables import initialize_all_variables

tf.logging.set_verbosity(tf.logging.INFO)


def create_DS(ds_num, v_pre, v_post, cleanse=False):
    ds1_files = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230', '107', '217']
    #ds1_files = ['101', '106']
    ds2_files = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234', '102', '104']
    #ds2_files = ['100', '103']
    freq = 360
    preX = v_pre
    postX = v_post
    dfall = {} 
    dfann = {} 
    dfseg = {} 
    dfseg1 = {}
    dfseg2 = {}
    dfseg1_fir = {}
    dfseg2_fir = {}
    dfseg1_baseline_values = {}
    dfseg2_baseline_values = {}
    segment_data = []
    segment_labels = []
    if (ds_num == "1"):
        ds_list = ds1_files;
    else:
        ds_list = ds2_files;
    
    # Load the necessary patient inputs    
    for patient_num in ds_list:
        dfall[patient_num] = pd.read_csv('../data/DS' + ds_num + '/' + patient_num + '_ALL_samples.csv', sep=',', header=0, squeeze=False)
        dfann[patient_num] = pd.read_csv('../data/DS' + ds_num + '/' + patient_num + '_ALL_ANN.csv', sep=',', header=0, parse_dates=[0], squeeze=False)
   
   
    # Butterworth filter: x -> y
    lowcut=0.01
    highcut=15.0
    signal_freq=360
    filter_order=1
    nyquist_freq = 0.5*signal_freq
    low=lowcut/nyquist_freq
    high=highcut/nyquist_freq
    b, a = signal.butter(filter_order, [low,high], btype="band")
                         
    # Standardize the beat annotations 
    # vals_to_replace = {'N':'N','L':'N','e':'N','j':'N','R':'N','A':'SVEB','a':'SVEB','J':'SVEB','S':'SVEB','V':'VEB','E':'VEB','F':'F','Q':'Q','P':'Q','f':'Q','U':'Q'}
    # use integers 0..4 instead of annotation...
    vals_to_replace = {'N':0, 'L':0, 'e':0, 'j':0, 'R':0, 'A':1, 'a':1, 'J':1, 'S':1, 'V':2, 'E':2, 'F':3, 'Q':4, 'P':4, 'f':4, 'U':4}
    
    for patient_num in ds_list:
        dfann[patient_num]['Type'] = dfann[patient_num]['Type'].map(vals_to_replace)    
        dfann[patient_num]['RRI'] = (dfann[patient_num]['sample'] - dfann[patient_num]['sample'].shift(1)) / 360
        dfann[patient_num] = dfann[patient_num][1:]  
    
    for patient_num in ds_list:
        annList = [];
        rriList = [];
        begNList = [];
        endNList = [];
        mixNList = [];
        sliceNList = [];

        for index, row in dfann[patient_num].iterrows():
            if (row['Type'] == 1) or (row['Type'] == 3) or (row['Type'] == 4):
                for i in (1,500):
                    Nbegin = row['sample'] - preX;
                    Nend = row['sample'] + postX;
                    begNList.append(Nbegin);
                    endNList.append(Nend);
                    annList.append(row['Type'])
                    rriList.append(row['RRI'])
                    mixNList = tuple(zip(begNList, endNList, annList, rriList))
            if (row['Type'] == 2):
                for i in (1,5):
                    Nbegin = row['sample'] - preX;
                    Nend = row['sample'] + postX;
                    begNList.append(Nbegin);
                    endNList.append(Nend);
                    annList.append(row['Type'])
                    rriList.append(row['RRI'])
                    mixNList = tuple(zip(begNList, endNList, annList, rriList))
        
            Nbegin = row['sample'] - preX;
            Nend = row['sample'] + postX;
            begNList.append(Nbegin);
            endNList.append(Nend);
            annList.append(row['Type'])
            rriList.append(row['RRI'])     
                     

        mixNList = tuple(zip(begNList, endNList, annList, rriList))
        
        for row in mixNList:
            dfseg = dfall[patient_num][(dfall[patient_num]['sample'] >= row[0]) & (dfall[patient_num]['sample'] <= row[1])]
            dfseg1 = dfseg[dfseg.columns[1:2]]
            dfseg2 = dfseg[dfseg.columns[2:3]]
            if (cleanse == True):
                dfseg1_fir = signal.lfilter(b, a, dfseg[dfseg.columns[1:2]])
                dfseg2_fir = signal.lfilter(b, a, dfseg[dfseg.columns[2:3]])
                dfseg1_baseline_values = peakutils.baseline(dfseg1_fir)
                dfseg2_baseline_values = peakutils.baseline(dfseg2_fir)
                dfseg1 = dfseg1_fir-dfseg1_baseline_values
                dfseg2 = dfseg2_fir-dfseg2_baseline_values
            training_inputs1 = np.asarray(dfseg1.flatten(), dtype=np.float32)
            training_inputs2 = np.asarray(dfseg2.flatten(), dtype=np.float32)
            training_inputs1 = np.concatenate((training_inputs1, np.asarray([row[3]], dtype=np.float32)))
            training_inputs2 = np.concatenate((training_inputs2, np.asarray([row[3]], dtype=np.float32)))
            segment_data.append(np.concatenate((training_inputs1, training_inputs2), axis=0))
            training_labels = row[2]
            segment_labels.append(training_labels)    
            
    segment_data = np.asarray(segment_data)
   
    return dfall, dfann, segment_data, segment_labels


def cnn_model_fn2(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    
    segment_data = tf.placeholder('float32', [None, 480])
    train_data = tf.placeholder('float32', [None, 480])
    eval_data = tf.placeholder('float32', [None, 480])
    x = tf.placeholder('float32', [None, 480])
    input_layer = tf.placeholder('float32', [None, 480])
    
    segment_labels = tf.placeholder('int32')
    train_labels = tf.placeholder('int32')
    eval_labels = tf.placeholder('int32')
    y = tf.placeholder('int32')

    input_layer = tf.reshape(features["x"], [-1, 1, 480, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 1, 480, 1]
    # Output Tensor Shape: [batch_size, 1, 478, 5]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=5,
        kernel_size=[1, 3],
        # kernel_initializer=,
        padding='valid',
        activation=tf.nn.leaky_relu)

    # print("conv1: ")
    # print(conv1.shape)
 
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 1, 478, 5]
    # Output Tensor Shape: [batch_size, 1, 239, 5]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=2)

    # print("pool1: ")
    # print(pool1.shape)
    
    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 1, 239, 5]
    # Output Tensor Shape: [batch_size, 1, 236, 10]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=10,
        kernel_size=[1, 4],
        # kernel_initializer="c2",
        # padding="same",
        activation=tf.nn.leaky_relu)

    # print("conv2: ")
    # print(conv2.shape)
    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 1, 236, 10]
    # Output Tensor Shape: [batch_size, 1, 118, 10]

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=2)

  # Convolutional Layer #3
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 1, 118, 10]
    # Output Tensor Shape: [batch_size, 1, 116, 20]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=20,
        kernel_size=[1, 3],
        # kernel_initializer=,
        padding='valid',
        activation=tf.nn.leaky_relu)

    # print("conv1: ")
    # print(conv1.shape)
 
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 1, 116, 20]
    # Output Tensor Shape: [batch_size, 1, 58, 20]
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1, 2], strides=2)
    
    # print("pool2: ")
    # print(pool2.shape)
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 1, 58, 20]
    # Output Tensor Shape: [batch_size, 1, 58, 20]
    pool3_flat = tf.reshape(pool3, [-1, 1 * 58 * 20])

    # print("pool2_flat: ")
    # print(pool2_flat.shape)
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense1 = tf.layers.dense(inputs=pool3_flat, units=30, activation=tf.nn.leaky_relu)

    # print("dense: ")
    # print(dense.shape)
    dense2 = tf.layers.dense(inputs=dense1, units=20, activation=tf.nn.leaky_relu)
    
    # Add dropout operation; 0.7 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense2, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    # print("dropout: ")
    # print(dropout.shape)
    
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=5)
    
    # print("logits: ")
    # print(logits.shape)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #con = tf.confusion_matrix(labels=labels, predictions=predictions["classes"])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
        }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  
def cnn_model_fn2_rri(features, labels, mode):
    """Model function for CNN."""
    # Input Layer

    segment_data = tf.placeholder('float32', [None, 482])
    train_data = tf.placeholder('float32', [None, 482])
    eval_data = tf.placeholder('float32', [None, 482])
    x = tf.placeholder('float32', [None, 482])
    input_layer = tf.placeholder('float32', [None, 482])
    
    segment_labels = tf.placeholder('int32')
    train_labels = tf.placeholder('int32')
    eval_labels = tf.placeholder('int32')
    y = tf.placeholder('int32')

    # print(np.unique(train_labels, return_counts=True))
    # print(np.unique(eval_labels, return_counts=True))

    input_layer = tf.reshape(features["x"], [-1, 1, 482, 1])

    # Convolutional Layer #1
    # Input Tensor Shape: [batch_size, 1, 482, 1]
    # Output Tensor Shape: [batch_size, 1, 478, 5]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=5,
        kernel_size=[1, 5],
        # kernel_initializer=,
        padding='valid',
        activation=tf.nn.leaky_relu)

 
    # Pooling Layer #1
    # Input Tensor Shape: [batch_size, 1, 478, 5]
    # Output Tensor Shape: [batch_size, 1, 239, 5]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=2)

    # Convolutional Layer #2
    # Input Tensor Shape: [batch_size, 1, 239, 5]
    # Output Tensor Shape: [batch_size, 1, 236, 10]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=10,
        kernel_size=[1, 4],
        # kernel_initializer="c2",
        # padding="same",
        activation=tf.nn.leaky_relu)

    # Pooling Layer #2
    # Input Tensor Shape: [batch_size, 1, 236, 10]
    # Output Tensor Shape: [batch_size, 1, 118, 10]

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=2)

  # Convolutional Layer #3
    # Input Tensor Shape: [batch_size, 1, 118, 10]
    # Output Tensor Shape: [batch_size, 1, 116, 20]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=20,
        kernel_size=[1, 3],
        # kernel_initializer=,
        padding='valid',
        activation=tf.nn.leaky_relu)

 
    # Pooling Layer #1
    # Input Tensor Shape: [batch_size, 1, 116, 20]
    # Output Tensor Shape: [batch_size, 1, 58, 20]
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1, 2], strides=2)
    
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 1, 58, 20]
    # Output Tensor Shape: [batch_size, 1, 58, 20]
    pool3_flat = tf.reshape(pool3, [-1, 1 * 58 * 20])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense1 = tf.layers.dense(inputs=pool3_flat, units=30, activation=tf.nn.leaky_relu)

    dense2 = tf.layers.dense(inputs=dense1, units=20, activation=tf.nn.leaky_relu)
    
    # Add dropout operation; 0.7 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense2, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=5)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #con = tf.confusion_matrix(labels=labels, predictions=predictions["classes"])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
        }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  
  
def main(unused_argv):
    """
    segment_data = tf.placeholder('float32', [None, 80])
    train_data = tf.placeholder('float32', [None, 80])
    eval_data = tf.placeholder('float32', [None, 80])
    x = tf.placeholder('float32', [None, 80])
    input_layer = tf.placeholder('float32', [None, 80])
    
    segment_labels = tf.placeholder('int32')
    train_labels = tf.placeholder('int32')
    eval_labels = tf.placeholder('int32')
    y = tf.placeholder('int32')
    """
    preX = 89
    postX = 150
    ds1_all, ds1_ann, ds1_seg, ds1_lab = create_DS("1",preX,postX,cleanse=True)
    ds2_all, ds2_ann, ds2_seg, ds2_lab = create_DS("2",preX,postX,cleanse=True)
    
    train_data = ds1_seg
    train_labels = np.asarray(ds1_lab, dtype=np.int32)
    eval_data = ds2_seg
    eval_labels = np.array(ds2_lab, dtype=np.int32)

    #print(np.unique(train_labels, return_counts=True))
    #print(np.unique(eval_labels, return_counts=True))

     # Create the Estimator
    ecg_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn2_rri, model_dir="/tmp/ecg_convnet_model2_rri")
    
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True)
    ecg_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
    
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=20,
        shuffle=False)
    eval_results = ecg_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    
if __name__ == "__main__":
    with tf.Session() as sess:
        tf.global_variables_initializer()
        tf.app.run()
    
