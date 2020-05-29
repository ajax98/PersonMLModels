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





# To find out which devices your operations and tensors are assigned to

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# # Add a channels dimension
# x_train = x_train[..., tf.newaxis]
# x_test = x_test[..., tf.newaxis]

# train_ds = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10000).batch(32)

# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



# Create an instance of the model
model = MobileNetLite()
# model.load_weights('192x192ckpts_tflite_replica/cp-0011.ckpt')


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=.045, beta_1=.98)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


data = DataLoader(64, image_width=96, image_height=96)
train_loader = data.generate_batch('train')
validation_loader = data.generate_batch('val')


# X_batch, y_batch = iter(train_loader).next()
# print("X batch shape: ", X_batch.shape)
# X_batch = X_batch.permute(0,2,3,1)
# X_batch = np.array(X_batch)
# X_batch = X_batch * 255
# print("X batch shape after modification: ", X_batch.shape)
# print("y_batch shape: ", y_batch.shape)


# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# current_time = '20200329-173852'
# train_log_dir = 'logs/gradient_tape/192/' + current_time + '/train'
# test_log_dir = 'logs/gradient_tape/192/' + current_time + '/test'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# X_batch = np.random.random((1,96,96,1))
# y_batch = np.array([1])

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    # tf.print("Predictions: ", predictions)
    loss = loss_object(labels, predictions)
    # print("Loss: ", loss)
    # print("Model Trainable : ", model.trainable_variables)
  gradients = tape.gradient(loss, model.trainable_variables)
  # tf.print("Gradient shape: ", len(gradients))
  # print("Model trainable variables: ", model.trainable_variables)
  # tf.print("Gradients : ", gradients)
  # print("Model Trainable Variables: ", model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


  train_loss(loss)
  train_accuracy(labels, predictions)
  # tf.print("Train Loss: ", train_loss.result())
  # tf.print("Train Accuracy: ", train_accuracy.result()*100)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
  # tf.print("Validation Loss: ", train_loss.result())
  # tf.print("Validation Accuracy: ", train_accuracy.result()*100)

# for i in range(1000):
#   train_step(X_batch, y_batch)

# print("X batch 0 shape", np.expand_dims(X_batch[0], axis=0).shape)

# x = np.random.random((1,96,96,1))
# model._set_inputs(x)
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# quantized_model = converter.convert()
# open("draft_model_1.tflite", "wb").write(quantized_model)

EPOCHS = 90

checkpoint_path = "120x120ckpts_tflite_replica/cp-{epoch:04d}.ckpt"
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

  # with train_summary_writer.as_default():
   
  #   tf.summary.scalar('loss', train_loss.result(), step=epoch)
  #   tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
    # tf.summary.trace_export('train_graph', step=epoch, profiler_outdir=train_log_dir)


  # tf.summary.trace_on(graph=True, profiler=True)
  for i, (test_images, test_labels) in enumerate(validation_loader):
    test_images = test_images.permute(0,2,3,1)
    test_images = np.array(test_images)
    test_images = test_images * 255
    test_labels = np.array(test_labels)
    
    test_step(test_images, test_labels)

    if(i % 10 == 0):
      tf.print("Batch Test Loss: ", test_loss.result())
      tf.print("Batch Test Accuracy: ", test_accuracy.result()*100)

  # with test_summary_writer.as_default():
    
  #   tf.summary.scalar('loss', test_loss.result(), step=epoch)
  #   tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
    # tf.summary.trace_export('test_graph', step=epoch, profiler_outdir=test_log_dir)
    
  model.save_weights(checkpoint_path.format(epoch=epoch))

  if(test_accuracy.result() > best_validation_accuracy):
    for file in glob.glob("120x120ckpts_tflite_replica/cp-{epoch:04d}*".format(epoch=epoch)):
      if("index" in file):
        shutil.copy(file, '120x120ckpts_tflite_replica/best.ckpt.index')
      if("data-00001-of-00002" in file):
        shutil.copy(file, '120x120ckpts_tflite_replica/best.ckpt.data-00001-of-00002')
      if("data-00000-of-00002" in file):
        shutil.copy(file, '120x120ckpts_tflite_replica/best.ckpt.data-00000-of-00002')

    best_validation_accuracy = test_accuracy.result()

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))


# print("ATTEMPTING TO LOAD WEIGHTS")

# model.load_weights('ckpts/best.ckpt')
# print("LOADED WEIGHTS")
