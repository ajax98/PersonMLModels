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
    # self.c1 = keras.layers.Conv2D(8, kernel_size=(3,3), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
 
    # # #MODEL LAYER 2
    # self.dc2 = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same',activation='relu', use_bias=False)
    # self.c2 = keras.layers.Conv2D(16, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    # # #MODEL LAYER 3
    # self.dc3 = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', use_bias=False)
    # self.c3 = keras.layers.Conv2D(32, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    # # # #MODEL LAYER 4
    # self.dc4 = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', use_bias=False)
    # self.c4 = keras.layers.Conv2D(32, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    # # #MODEL LAYER 5
    # self.dc5 = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', use_bias=False)
    # self.c5 = keras.layers.Conv2D(64, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    # # # #MODEL LAYER 6
    # self.dc6 = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', use_bias=False)
    # self.c6 = keras.layers.Conv2D(64, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    # #MODEL LAYER 7
    # self.dc7 = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', use_bias=False)
    # self.c7 = keras.layers.Conv2D(128, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    # # #MODEL LAYER 8
    # self.dc8 = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', use_bias=False)
    # self.c8 = keras.layers.Conv2D(128, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    # # #MODEL LAYER 9
    # self.dc9 = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', use_bias=False)
    # self.c9 =keras.layers.Conv2D(128, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    # # #MODEL LAYER 10
    # self.dc10 = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', use_bias=False)
    # self.c10 = keras.layers.Conv2D(128, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    # #MODEL LAYER 11
    # self.dc11 = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', use_bias=False)
    # self.c11 = keras.layers.Conv2D(128, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    # #MODEL LAYER 12
    # self.dc12 =keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', use_bias=False)
    # self.c12 = keras.layers.Conv2D(128, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    # #MODEL LAYER 13
    # self.dc13 = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', use_bias=False)
    # self.c13 = keras.layers.Conv2D(256, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    # #MODEL LAYER 14
    # self.dc14 = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', use_bias=False)
    # self.c14 = keras.layers.Conv2D(256, kernel_size=(1,1), strides=(1, 1), padding='valid', activation='relu', use_bias=False)
    #Layer 1
    # self.conv_bn1 = ConvBn(4, (2,2))
    # #Layer 2
    # self.dp1 = ConvDP(16, (1,1))
    # #Layer 3
    # self.dp2 = ConvDP(32, (2,2))
    # #Layer 4
    # self.dp3 = ConvDP(32, (1,1))
    # #Layer 5
    # self.dp4 = ConvDP(64, (2,2))

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
    # self.layer16 = keras.layers.Conv2D(2, kernel_size=(3,3), strides=(1, 1), padding='same', use_bias=False )
    # self.layer17 = keras.layers.BatchNormalization()
    # self.layer18 = keras.layers.ReLU()
    # self.layer19 = keras.layers.Reshape((1,2))


    # #Layer 6
    # self.dp5 = ConvDP(64, (1,1))
    # #Layer 7
    # self.dp6 = ConvDP(128, (2,2))
    # #Layer 8
    # self.dp7 = ConvDP(128, (1,1))
    # #Layer 9
    # self.dp8 = ConvDP(128, (1,1))
    # #Layer 10
    # self.dp9 = ConvDP(128, (1,1))
    # #Layer 11
    # self.dp10 = ConvDP(128, (1,1))
    # #Layer 12
    # self.dp11 = ConvDP(128, (1,1))
    # #Layer 13
    # self.dp12 = ConvDP(256, (2,2))
    # #Layer 14
    # self.dp13 = ConvDP(256, (1,1))
    # self.ap = keras.layers.AveragePooling2D(pool_size=(3, 3))
    # self.flatten = keras.layers.Flatten()
    # self.dense = keras.layers.Dense(2)

  def call(self, inputs):
  	# relu = keras.layers.ReLU()
    # print("Input: ", inputs)
    #x1 = (self.conv_bn1(inputs))
    # tf.print("x1 shape: ", x1.shape)
    #x2 = (self.dp1(x1))
    # tf.print("x2 shape: ", x2.shape)
    #x3 = (self.dp2(x2))
    # tf.print("x3 shape: ", x3.shape)
    #x4 = (self.dp3(x3))
    # tf.print("x4 shape: ", x4.shape)
    #x5 = (self.dp4(x4))
    #tf.print("x5 shape: ", x5.shape)
    # x6 = (self.dp5(x5))
    # #tf.print("x6 shape: ", x6.shape)
    # x7 = (self.dp6(x6))
    # #tf.print("x7 shape: ", x7.shape)
    # x8 = (self.dp7(x7))
    # #tf.print("x8 shape: ", x8.shape)
    # x9 = (self.dp8(x8))
    # #tf.print("x9 shape: ", x9.shape)
    # x10 = (self.dp9(x9))
    # #tf.print("x10 shape: ", x10.shape)
    # x11 = (self.dp10(x10))
    #tf.print("x11 shape: ", x11.shape)
    # x12 = (self.dp11(x11))
    # #tf.print("x12 shape: ", x12.shape)
    # x13 = (self.dp12(x12))
    # #tf.print("x13 shape: ", x13.shape)
    # x14 = (self.dp13(x13))
    # #tf.print("x14 shape: ", x14.shape)
    
    #fc = self.ap(x5)
    #tf.print("fc shape: ", fc.shape)
    #flattened_fc = self.flatten(fc)
    #tf.print("flattened fc shape: ", flattened_fc.shape)

    #logits = self.dense(flattened_fc)
    #tf.print("logits: ", logits.shape)
    # tf.print("TFLITE BABY")
    # tf.print("input shape: ", inputs.shape)
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
    # x18 = self.layer18(x17)

    # x19 = self.layer19(x18)
    # tf.print("x18 shape: ", x18.shape)
    # x19 = self.layer19(x18)
    # tf.print("x19 shape: ", x19.shape)
    # x20 = self.layer20(x19)
    # tf.print("output shape: ", x20.shape)

    # dense = self.layer19(self.layer18(self.layer16(x15)))

    return x17

