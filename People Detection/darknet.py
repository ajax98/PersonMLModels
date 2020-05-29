from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416, debug=False):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.debug = debug
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            if self.debug:
            	print('Total Loss: %f' % total_loss, 'Loss x: %f' % loss_x.data, 'Loss y: %f' % loss_y.data, 'Loss w: %f' % loss_w.data, 'Loss h: %f' % loss_h.data, 'Loss conf obj: %f' % loss_conf_obj.data, 'Loss conf no obj: %f' % loss_conf_noobj.data, 'Loss conf : %f' % loss_conf.data, 'Loss cls: %f' %loss_cls.data)
            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class DarkNet(nn.Module):

	def __init__(self, debug=False):
		super(DarkNet, self).__init__()

		self.bn1 = nn.BatchNorm2d(16, momentum=0.9, eps=1e-5)
		self.bn2 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-5)
		self.bn3 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)
		self.bn4 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)
		self.bn5 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)
		self.bn6 = nn.BatchNorm2d(512, momentum=0.9, eps=1e-5)
		self.bn7 = nn.BatchNorm2d(1024, momentum=0.9, eps=1e-5)

		#pathway 1
		self.bn8p1 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)
		self.bn9p1 = nn.BatchNorm2d(512, momentum=0.9, eps=1e-5)
		self.bn10p1 = nn.BatchNorm2d(18, momentum=0.9, eps=1e-5)

		#pathway2
		self.bn8p2 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)
		self.bn9p2 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)


		self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size = 3, stride=1, padding=1, bias=False),
								   self.bn1,
								   nn.LeakyReLU(.1),
								   
				
				  				  )

		self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size = 3, stride=1, padding=1, bias=False),
								   self.bn2,
								   nn.LeakyReLU(.1),
				
				  				  )

		self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, stride=1, padding=1, bias=False),
								   self.bn3,
								   nn.LeakyReLU(.1),
				
				  				  )

		self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size = 3, stride=1, padding=1, bias=False),
								   self.bn4,
								   nn.LeakyReLU(.1),
				
				  				  )

		self.conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 3, stride=1, padding=1, bias=False),
								   self.bn5,
								   nn.LeakyReLU(.1),
				
				  				  )

		self.conv6 = nn.Sequential(nn.Conv2d(256, 512, kernel_size = 3, stride=1, padding=1, bias=False),
								   self.bn6,
								   nn.LeakyReLU(.1),
				
				  				  )
		self.zeropad = nn.ZeroPad2d(padding=(0,1,0,1))


		self.conv7 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size = 3, stride=1, padding=1, bias=False),
								   self.bn7,
								   nn.LeakyReLU(.1),
				  				  )


		#pathway one

		self.conv8p1 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size = 1, stride=1, bias=False),
								   self.bn8p1,
								   nn.LeakyReLU(.1),
				  				  )
		self.conv9p1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size = 3, stride=1, padding=1, bias=False),
								   self.bn9p1,
								   nn.LeakyReLU(.1),
				  				  )
		self.conv10p1 = nn.Sequential(nn.Conv2d(512, 18, kernel_size = 1, stride=1, bias=False),
								 	  )

		self.yolo1 = YOLOLayer(anchors=[(81,82), (135,169), (344,319)], num_classes=1, debug=debug)
		# self.yolo1 = YOLOLayer(anchors=[(81,82)], num_classes=1, debug=debug)


		#pathway two

		self.conv8p2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size = 1, stride=1, bias=False),
								   self.bn8p2,
								   nn.LeakyReLU(.1),
				  				  )

		self.conv9p2 = nn.Sequential(nn.Conv2d(384, 256, kernel_size = 3, stride=1, padding=1, bias=False),
								   self.bn9p2,
								   nn.LeakyReLU(.1),
				  				  )

		self.conv10p2 = nn.Sequential(nn.Conv2d(256, 18, kernel_size = 1, stride=1, padding=1, bias=False),
				  				  )

		self.yolo2 = YOLOLayer(anchors=[(23,27), (37,58), (81,82)], num_classes=1)


	def forward(self, x, targets=None):

		yolo_outputs = []
		loss = 0
		img_dim = x.shape[2]
		# print("Start: ", x.shape)
		x = self.conv1(x)
		# print("Conv 1: ", x.shape)
		x = nn.MaxPool2d(kernel_size = 2, stride = 2)(x)
		# print("After Maxpool: ", x.shape)
		x = self.conv2(x)
		# print("Conv 2: ", x.shape)
		x = nn.MaxPool2d(kernel_size = 2, stride = 2)(x)
		# print("After Maxpool: ", x.shape)
		x = self.conv3(x)
		# print("Conv 3: ", x.shape)
		x = nn.MaxPool2d(kernel_size = 2, stride = 2)(x)
		# print("After Maxpool: ", x.shape)
		x = self.conv4(x)
		# print("Conv 4: ", x.shape)
		x = nn.MaxPool2d(kernel_size = 2, stride = 2)(x)
		# print("After Maxpool: ", x.shape)
		x = self.conv5(x)
		x_f128 = x
		# print("Conv 5: ", x.shape)
		x = nn.MaxPool2d(kernel_size = 2, stride = 2)(x)
		# print("After Maxpool: ", x.shape)
		x = self.conv6(x)
		# print("Conv 6: ", x.shape)
		x = self.zeropad(x)
		# print("Conv 6 After Zero Pad: ", x.shape)
		x = nn.MaxPool2d(kernel_size = 2, stride = 1)(x)
		# print("After Maxpool: ", x.shape)
		x = self.conv7(x)
		# print("Conv 7: ", x.shape)
		

		#pathway 1
		x = self.conv8p1(x)
		x_f256 = x
		# print("Conv 8 P1: ", x.shape)
		x = self.conv9p1(x)
		# print("Conv 9 P1: ", x.shape)
		x = self.conv10p1(x)
		self.yolo1_inp = x
		# print("Conv 10 P1: ", x.shape, targets.shape)
		x, layer_loss = self.yolo1(x, targets, img_dim)
		loss += layer_loss
		yolo_outputs.append(x)
		# print("YOLO OUTPUT: ", x)

		#pathway 2
		x = self.conv8p2(x_f256)
		# print("Conv 8 P2: ", x.shape)
		x =  torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
		# print("X Shape: ", x.shape)
		# print("X_f128 Shape: ", x_f128.shape)
		# p2d = (1, 1, 1, 1)
		# x_f128 = F.pad(x_f128, p2d, 'constant', 0)
		# print("X_f128 Shape After Padding: ", x_f128.shape)
		x = torch.cat((x, x_f128), dim=1)
		x = self.conv9p2(x)
		# print("Conv 9 P2: ", x.shape)
		x = self.conv10p2(x)
		# print("Conv 10 P2: ", x.shape, targets.shape)
		x, layer_loss = self.yolo2(x, targets, img_dim)
		loss += layer_loss
		yolo_outputs.append(x)
		# print("YOLO OUTPUT: ", x)

		yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
		# print("YOLO OUTPUTS:", yolo_outputs)
		return yolo_outputs if targets is None else (loss, yolo_outputs)















