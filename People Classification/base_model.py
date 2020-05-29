import tensorflow as tf
from tensorflow import keras

class ConvBn(tf.keras.layers.Layer):
  def __init__(self, oup, strides):
    super(ConvBn, self).__init__()
    self.oup = oup
    self.strides = strides
    self.conv = keras.layers.Conv2D(oup, kernel_size=(3,3), strides=self.strides, padding='same', use_bias=False)
    self.bn = keras.layers.BatchNormalization()
  
  def call(self, inp):
    relu = keras.layers.ReLU()
    conv_output = self.conv(inp)
    bn_output = self.bn(conv_output)
    return relu(bn_output)

class ConvDP(tf.keras.layers.Layer):
  def __init__(self, oup, strides):
    super(ConvDP, self).__init__()
    self.oup = oup
    self.strides = strides
    self.dpconv = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=self.strides, padding='same', use_bias=False)
    self.bn1 = keras.layers.BatchNormalization()
    self.conv = keras.layers.Conv2D(oup, kernel_size=(1,1), strides=(1, 1), padding='valid', use_bias=False)
    self.bn2 = keras.layers.BatchNormalization()
  def call(self, inp):
    relu = keras.layers.ReLU()
    conv_output = self.dpconv(inp)
    bn_output = self.bn1(conv_output)
    relu_output = relu(bn_output)
    conv2_output = self.conv(relu_output)
    bn2_output = self.bn2(conv2_output)
    return relu(bn2_output)
    
    


class MobileNetLite(tf.keras.Model):

  def __init__(self):
    super(MobileNetLite, self).__init__()

    self.layer1 = ConvBn(8, (2,2))
    self.layer2 = ConvDP(16, (2,2))
    self.layer3 = ConvDP(32, (2,2))
    self.layer4 = ConvDP(32, (1,1))
    self.layer5 = ConvDP(64, (2,2))
    self.layer6 = ConvDP(64, (1,1))
    self.layer7 = ConvDP(128, (2,2))
    self.layer8 = ConvDP(128, (1,1))
    self.layer9 = ConvDP(128, (1,1))
    self.layer10 = ConvDP(128, (1,1))
    self.layer11 = ConvDP(128, (1,1))
    self.layer12 = ConvDP(128, (1,1))
    self.layer13 = ConvDP(256, (2,2))
    self.layer14 = ConvDP(256, (1,1))
    self.layer15 = keras.layers.AveragePooling2D(pool_size=(2,2))
    self.layer16 = keras.layers.Flatten()
    self.layer17 = keras.layers.Dense(2)

  def call(self, inputs):

    x1 = self.layer1(inputs)
    # tf.print("x1 shape: ", x1.shape)
    x2 = self.layer2(x1)
    # tf.print("x2 shape: ", x2.shape)
    x3 = self.layer3(x2)
    # tf.print("x3 shape: ", x3.shape)
    x4 = self.layer4(x3)
    # tf.print("x4 shape: ", x4.shape)
    x5 = self.layer5(x4)
    # tf.print("x5 shape: ", x5.shape)
    x6 = self.layer6(x5)
    # tf.print("x6 shape: ", x6.shape)
    x7 = self.layer7(x6)
    # tf.print("x7 shape: ", x7.shape)
    x8 = self.layer8(x7)
    # tf.print("x8 shape: ", x8.shape)
    x9 = self.layer9(x8)
    # tf.print("x9 shape: ", x9.shape)
    x10 = self.layer10(x9)
    # tf.print("x10 shape: ", x10.shape)
    x11 = self.layer11(x10)
    # tf.print("x11 shape: ", x11.shape)
    x12 = self.layer12(x11)
    # tf.print("x12 shape: ", x12.shape)
    x13 = self.layer13(x12)
    # tf.print("x13 shape: ", x13.shape)
    x14 = self.layer14(x13)
    # tf.print("x14 shape: ", x14.shape)
    x15 = self.layer15(x14)
    # tf.print("x15 shape: ", x15.shape)
    x16 = self.layer16(x15)
    # tf.print("x16 shape: ", x16.shape)
    x17 = self.layer17(x16)
    # tf.print("x17 shape: ", x17.shape)

    return x17

