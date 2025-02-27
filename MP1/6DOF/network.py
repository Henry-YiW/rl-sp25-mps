import torch
import torch.nn as nn
from torchvision import models
from utils import make_rotation_matrix
from absl import flags
FLAGS = flags.FLAGS


class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Identity()
        self.num_classes = num_classes
        num_outputs = self.num_classes + 9 + 3
        self.one = 1

        self.head = nn.Sequential(
            nn.Linear(512 + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

    def crop_batched_tensor(self,image_tensor, bbox):
        """
        Crops a batched image tensor based on the bounding box.

        Args:
            image_tensor (torch.Tensor): Image tensor of shape (B, C, H, W).
            bbox (tuple): Bounding box as (x, y, width, height).

        Returns:
            torch.Tensor: Cropped image tensor of shape (B, C, h, w).
        """
        x, y, w, h = bbox  # Bounding box (x, y, width, height)
    
        return image_tensor[:, :, y:y+h, x:x+w]  # Keep batch & channels

    def forward(self, image, bbox):
        # print(image.shape, bbox.shape)
        # image = self.crop_batched_tensor(image, bbox)
        # print(image.shape)
        x = self.resnet(image)
        x = torch.cat((x, bbox), dim=1)
        x = self.head(x)
        logits, R, t = torch.split(x, [self.num_classes, 9*self.one, 3*self.one], dim=1)
        return logits, R, t

    def process_output(self, outs):
        with torch.no_grad():
            logits, R, t = outs 
            cls = logits.argmax(dim=1)
            R = make_rotation_matrix(R.reshape(-1, 3, 3))
            t = t.reshape(-1, 3, 1)
            return cls, R, t
    
