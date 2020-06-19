# Person Classification

## Requirements
The code was run using: 
1. Python 3.6
2. Tensorflow 2.1.1
3. PyTorch 1.3.1
4. Pillow 5.0
5. VisualWakeWords (https://github.com/Mxbonn/visualwakewords)
6. Numpy
7. Matplotlib


## Train Models
To train the following models for person classification run :
`python3 ModelTrainer.py [-h] [--startpoint STARTPOINT] epochs image_width image_height batch_size checkpoint_path`

i.e.

`python3 ModelTrainer.py 90 96 96 32 ./checkpoint_path`

## Evaluate Models

To evaluate models for person classification run : 

`python3 ModelCheckpointEvaluate.py [-h] checkpoint image_width image_height batchsize`

i.e.

`python3 ModelCheckpointEvaluate.py 48x48_model/best.ckpt 48 48 32`

## Convert Models

To convert models to tflite for person classification run :

`python3 ModelTFLiteConverter.py [-h] num_calib_steps image_width image_heigh checkpoint output_name`

i.e.

`python3 ModelTFLiteConverter.py 100 96 96 96x96_model/best.ckpt tflite_models/96x96_new.tflite`


To convert tflite models to a cc file run:

`xxd -i converted_model.tflite > model_data.cc`

More information on this can be found on https://www.tensorflow.org/lite/microcontrollers/build_convert



