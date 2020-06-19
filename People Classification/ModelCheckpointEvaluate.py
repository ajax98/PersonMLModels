from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from data_loader_coco import DataLoader
from base_model import MobileNetLite
import shutil
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="Name of checkpoint to import", type=str)
parser.add_argument("image_width", help=" Input Image width", type=int)
parser.add_argument("image_height", help="Input Image height", type=int)
parser.add_argument("batchsize", help="Batchsize", type=int)


args = parser.parse_args()

model = MobileNetLite(args.image_height)
model.load_weights(args.checkpoint)	
data = DataLoader(args.batchsize, image_width=args.image_width, image_height=args.image_height)

validation_loader = data.generate_batch('val')
# train_loader = data.generate_batch('train')

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  #tf.print("Predictions: ", predictions)
  #tf.print("Label: ", labels)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

print("Starting validation of model!")
for i, (test_images, test_labels) in enumerate(validation_loader):
	test_images = test_images.permute(0,2,3,1)
	test_images = np.array(test_images)
	test_images = 255*test_images
	test_labels = np.array(test_labels)
	test_step(test_images, test_labels)
	if(i % 10 == 0):
		tf.print("Batch Test Loss: ", test_loss.result())
		tf.print("Batch Test Accuracy: ", test_accuracy.result()*100)

# tf.print("FINAL ACCURACY: ", test_accuracy.result()*100)
