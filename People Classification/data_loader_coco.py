import numpy as np 
import pyvww
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image




class DataLoader():

	def __init__(self, batch_size, image_width=96, image_height=96, num_channels=1, shuffle=False):
		
		self.batch_size = batch_size
		self.image_width = image_width
		self.image_height = image_height
		self.num_channels = num_channels

		self.preprocess = transforms.Compose([
			transforms.Grayscale(),
			transforms.Resize((self.image_width,self.image_height)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# transforms.Normalize([0.44747058], [0.11781323])
		])
		self.preprocess_val = transforms.Compose([
        	transforms.Grayscale(),
        	transforms.Resize((self.image_width,self.image_height)),
        	transforms.ToTensor(),
        	# transforms.Normalize([0.44747058], [0.11781323])
		])

		self.preprocess_val_simple = transforms.Compose([
        	transforms.Grayscale(),
        	transforms.Resize((self.image_width,self.image_height)),
        	transforms.ToTensor()
		])

		self.train_data = pyvww.pytorch.VisualWakeWordsClassification(root="/home/ajay/Desktop/MastersProject/PyTorchModel/coco/images", annFile="/home/ajay/Desktop/MastersProject/PyTorchModel/vww/annotations/instances_train.json", transform=self.preprocess)
		self.val_data = pyvww.pytorch.VisualWakeWordsClassification(root="/home/ajay/Desktop/MastersProject/PyTorchModel/coco/images", annFile="/home/ajay/Desktop/MastersProject/PyTorchModel/vww/annotations/instances_val.json", transform=self.preprocess_val)
		# self.val_data = pyvww.pytorch.VisualWakeWordsClassification(root="/home/ajay/Desktop/MastersProject/PyTorchModel/coco/images", annFile="/home/ajay/Desktop/MastersProject/PyTorchModel/vww/annotations/instances_val.json", transform=self.preprocess_val_simple)

		self.train_data_len = len(self.train_data)
		self.val_data_len = len(self.val_data)  
		self.shuffle = shuffle



	def load_data(self):
		return self.image_width, self.image_height, self.num_channels, self.train_data_len, self.val_data_len


	def generate_batch(self, type='train'):

		if (type == 'train'):
			train_loader = torch.utils.data.DataLoader(
						    self.train_data,
						    batch_size=self.batch_size, shuffle=self.shuffle,
						    num_workers=4, pin_memory=True)
			return train_loader
		elif (type == 'val'):
			 val_loader = torch.utils.data.DataLoader(
						    self.val_data,
						    batch_size=self.batch_size, shuffle=self.shuffle,
						    num_workers=4, pin_memory=True)
			 return val_loader
