# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import ml_helper as mlh
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='9'

import numpy as np
import tensorflow as tf

from pathlib import Path
import argparse
import time
import json
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from multiprocessing.pool import ThreadPool

app_path = os.getcwd()

dims = 299

def operation(sess, softmax, image, image_number):
    global arr_labels

    print('\tProcessing tile : ' + mlh.bcolors.OKGREEN + str(image) + mlh.bcolors.ENDC)

    im = tf.gfile.FastGFile(image, 'rb').read()

    prediction = sess.run(softmax, {'DecodeJpeg/contents:0': im})
    pred = np.argmax(prediction[0])

    _hash = mlh.find_between(image, 'IMAGES_TO_CHECK' + os.sep, os.sep + '_TILE_')

    if pred > -1:
        class_dir = app_path + os.sep + 'classification_results' + os.sep + 'IMAGES_TO_CHECK' + os.sep + _hash + os.sep + arr_labels[pred]
    else:
        class_dir = app_path + os.sep + 'classification_results' + os.sep + 'IMAGES_TO_CHECK' + os.sep + _hash + os.sep + 'undetected'


    if not os.path.exists(class_dir):
        os.mkdir(class_dir)

    newFile = class_dir + os.sep + _hash + '_IMG_X'+str(image_number)+'.jpg'

    if os.path.exists(newFile):
        os.remove(newFile)

    os.rename(image, newFile)

    return prediction, pred, image_number

def predict(image_file):
    from PIL import Image, ImageOps
    global dims
    global arr_labels
    global verbose
    global sess
    global softmax_tensor

    start_time = time.time()

    image_list = []
    with open(image_file) as f:
        image_list = f.read().splitlines()

    print('\t' + mlh.bcolors.OKGREEN + 'Found : ' + mlh.bcolors.OKYELLOW + str(len(image_list)) + mlh.bcolors.ENDC)

    pool = ThreadPool()

    threads = [pool.apply_async(operation, args=(sess, softmax_tensor, image_list[k], k,)) for
           k in range(len(image_list))]

    json_str = '['

    sum = 0

    result = []
    for thread in threads:
        result.append(thread.get())

    for i in result:
        #print(i)
        if not json_str == '[':
            json_str += ', '

        str_pred = "[["
        for x in i[0][0]:

            if not str_pred == "[[":
                str_pred += ", "
            str_pred += str(format(x, 'f'))

        str_pred += "]]"

        json_str += '{ "img" : "' + str(int(i[2])) + '", "prediction_array" : "' + str_pred + '", "result" : "'

        json_str += str(i[1]) + '"}'


    json_str += ']'

    text_file = open(image_file.replace('.txt', '.json'), "w")
    text_file.write(json_str)
    text_file.close()

    elapsed_time = time.time() - start_time

    print('Total : ' + str(len(image_list))
          + ' tiles in ' + str(round(elapsed_time,4)) + ' secs.\r\n')

    if verbose:
        print('JSON output')
        print(json.dumps(json_str, indent=4, sort_keys=True))


class MyHandler(PatternMatchingEventHandler):
    patterns = ["*.txt"]

    def process(self, event):
        global obs_folder

        time.sleep(2)
        print('Processing file : ' + mlh.bcolors.OKGREEN + str(event.src_path) + mlh.bcolors.ENDC)
        predict(event.src_path)

    def on_created(self, event):
        self.process(event)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--img", help = "Path to the image file with all the tiles.")
#ap.add_argument("-t", "--tfl", help = "Path to the TFL File.")
ap.add_argument("-v", "--verbose", help="Verbose output")
args = vars(ap.parse_args())


verbose = args.get("verbose", False)
#tfl_path = args.get("mask", False)
image = args["img"]

if verbose:
    print('Verbose Output. JSON Will be printed out')

classes = 0
arr_labels = []

with open('tf_files/retrained_labels.txt') as f:
    arr_labels = f.read().splitlines()

#arr_labels = ['uncapped_obstructed','trash','capped','uncapped']
#for subdir, dirs, files in os.walk('..//pipes_training_images' + os.sep + 'train'):
#    classes = len(dirs)
#    arr_labels = dirs
#    break

with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

print('Model loaded. Ready to process\r\n')
obs_folder = app_path + os.sep + 'classification_results' + os.sep + 'IMAGES_TO_CHECK'
print('Observing folder : ' + obs_folder)

observer = Observer()
observer.schedule(MyHandler(), path=obs_folder, recursive=True)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
