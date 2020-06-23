import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_image_width", help=" Input Image width", type=int)
parser.add_argument("input_image_height", help="Input Image height", type=int)
parser.add_argument("input_image_depth", help="Input Image depth", type=int)


args = parser.parse_args()

input_shape = (args.input_image_depth,args.input_image_width,args.input_image_height)


#Regular Convolution Macs = K × K × Cin × Hout × Wout × Cout
#Depthwise Convolution Macs =  K × K × Cin × Hout × Wout) + (Cin × Hout × Wout × Cout) or
# Cin × Hout × Wout × (K × K + Cout)
#Describe each convolution as (kernel size, h_s, w_s, cout, dp_bool)
info_array = [(3, 2, 2, 8, False),
              (3, 2, 2, 16, True),
              (3, 2, 2, 32, True),
              (3, 1, 1, 32, True),
              (3, 2, 2, 64, True),
              (3, 1, 1, 64, True),
              (3, 2, 2, 128, True),
              (3, 1, 1, 128, True),
              (3, 1, 1, 128, True),
              (3, 1, 1, 128, True),
              (3, 1, 1, 128, True),
              (3, 1, 1, 128, True),
              (3, 2, 2, 128, True),
              (3, 1, 1, 256, True)
            ]

current_layer_info = input_shape
MACCS = 0
for i in range(len(info_array)):
    out_layer_info = info_array[i]
    kernel_size, h_s, w_s, c_out, dp_bool = out_layer_info
    c_in, h_in, w_in = current_layer_info
    h_out = h_in/h_s
    w_out = w_in/w_s

    if dp_bool:
        MACCS += c_in * h_out * w_out * (kernel_size * kernel_size + c_out)
    else:
        MACCS += c_in * (kernel_size * kernel_size) * h_out * w_out * c_out

    current_layer_info = (c_out, h_out, w_out)

print("TOTAL MACCS: ", MACCS)

