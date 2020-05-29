from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import argparse
from data_loader_coco import DataLoader
import numpy as np

#Model importing, change when modifying model
from tflite_replica4 import MobileNetLite


parser = argparse.ArgumentParser()
parser.add_argument("num_calib_steps", help="Number of steps for callibration", type=int)
parser.add_argument("image_width", help=" Input Image width", type=int)
parser.add_argument("image_height", help="Input Image height", type=int)
parser.add_argument("checkpoint", help="Checkpoint to load from", type=str)
parser.add_argument("output_name", help="Output Tflite file name", type=str)

args = parser.parse_args()


data = DataLoader(args.num_calib_steps, image_width=args.image_width, image_height=args.image_height)
train_loader = data.generate_batch('train')
validation_loader = data.generate_batch('val')

X_batch, y_batch = iter(train_loader).next()
X_batch = X_batch.permute(0,2,3,1)
X_batch = np.array(X_batch)
X_batch = X_batch*255
y_batch = np.array(y_batch)


#Define relevant dataset
def representative_dataset_gen():
  for i in range(args.num_calib_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [np.expand_dims(X_batch[i], axis=0)]

print("X batch shape: ", X_batch.shape)
print("y batch shape: ", y_batch.shape)


model = MobileNetLite()
model.load_weights(args.checkpoint)

#model inputs change if dimensions change
x = np.random.random((1,args.image_height,args.image_width,1))
print("X: ", x)
print("Input shape: ", x.shape)
model._set_inputs(x)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.inference_type = tf.uint8
converter.experimental_new_converter=False
quantized_model = converter.convert()
open(args.output_name, "wb").write(quantized_model)




