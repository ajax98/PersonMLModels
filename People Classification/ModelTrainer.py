from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from data_loader_coco import DataLoader
from base_model import MobileNetLite
import shutil
import glob
import datetime
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("epochs", help="Number of epochs", type=int)
parser.add_argument("image_width", help=" Input Image width", type=int)
parser.add_argument("image_height", help="Input Image height", type=int)
parser.add_argument("batch_size", help="Input Batch Size", type=int)
parser.add_argument("checkpoint_path", help="Checkpoint path to save to ", type=str)
parser.add_argument("--startpoint", help="Load previously trained weights to speed up training", default=True)
args = parser.parse_args()

#Variables epochs, checkpoint directory, size of images, 

# Create an instance of the model
model = MobileNetLite(args.image_height)
if(args.startpoint == True):
  model.load_weights('./startpoint.ckpt')




#Define loss for training
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#Set optimizer 
optimizer = tf.keras.optimizers.Adam(learning_rate=.045, beta_1=.98)
# optimizer = tf.keras.optimizers.Adam(learning_rate=.1, beta_1=.98)


#Create metrics for train and test loss and accuracies
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#Load data of specified image width and height
data = DataLoader(args.batch_size, image_width=args.image_width, image_height=args.image_height)

train_loader = data.generate_batch('train')
validation_loader = data.generate_batch('val')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  # tf.print("Gradients: ", gradients)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
  # tf.print("Train Loss: ", train_loss.result())
  # tf.print("Train Accuracy: ", train_accuracy.result()*100)


@tf.function
def test_step(images, labels):

  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)



EPOCHS = args.epochs

checkpoint_path = args.checkpoint_path+"cp-{epoch:04d}.ckpt"
best_validation_accuracy = 0

for epoch in range(0, EPOCHS):


  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  # tf.summary.trace_on(graph=True, profiler=True)
  for i, (train_images, train_labels) in enumerate(train_loader): 
    train_images = train_images.permute(0,2,3,1)
    train_images = np.array(train_images)
    train_images = train_images * 255
    train_labels = np.array(train_labels)
    
    train_step(train_images, train_labels)
  
    if(i % 10 == 0):
      tf.print("Batch Train Loss: ", train_loss.result())
      tf.print("Batch Train Accuracy: ", train_accuracy.result()*100)


  for i, (test_images, test_labels) in enumerate(validation_loader):
    test_images = test_images.permute(0,2,3,1)
    test_images = np.array(test_images)
    test_images = test_images * 255
    test_labels = np.array(test_labels)
    
    test_step(test_images, test_labels)

    if(i % 10 == 0):
      tf.print("Batch Test Loss: ", test_loss.result())
      tf.print("Batch Test Accuracy: ", test_accuracy.result()*100)

  
  model.save_weights(checkpoint_path.format(epoch=epoch))

  if(test_accuracy.result() > best_validation_accuracy):
    for file in glob.glob((args.checkpoint_path+"cp-{epoch:04d}*").format(epoch=epoch)):
      if("index" in file):
        shutil.copy(file, (args.checkpoint_path+'best.ckpt.index'))
      if("data-00001-of-00002" in file):
        shutil.copy(file, args.checkpoint_path+'best.ckpt.data-00001-of-00002')
      if("data-00000-of-00002" in file):
        shutil.copy(file, args.checkpoint_path+'best.ckpt.data-00000-of-00002')

    best_validation_accuracy = test_accuracy.result()

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))



