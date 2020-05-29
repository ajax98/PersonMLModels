import imageio
import argparse
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("InFile", help="image input file name", type=str)
parser.add_argument("OutFile", help="Raw image out file name", type=str)
parser.add_argument("ResizeHeight", help="Resize height", type=int)
parser.add_argument("ResizeWidth", help="Resize width", type=int)
parser.add_argument("Grayscale", help="Grayscale or not", type=bool)


args = parser.parse_args()
print("Filename: ", args.InFile);

# print("args outfile: ", args.OutFile)
f = open(args.OutFile, "w");

# im = imageio.imread(args.InFile)  # read a standard image
im = Image.open(args.InFile)
resize = transforms.Resize((args.ResizeHeight,args.ResizeWidth))
im = resize(im)
if args.Grayscale:
	grayscale = transforms.Grayscale()
	im = grayscale(im)
im = np.array(im)


print(" Shape of image: ", im.shape)  # im is a numpy array

lineCount = 0
f.write("        ")

for  i in range(im.shape[0]) : 
   for j in range(im.shape[1]) :
   	# print(i, j, im[i,j])
   	f.write(hex(im[i,j]))
   	f.write(", ")
   	if lineCount > 15 :
   		f.write("\n")
   		f.write("        ")
   		lineCount = 0
   	lineCount += 1

f.close()


