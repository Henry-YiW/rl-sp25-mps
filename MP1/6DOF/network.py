import torch
import torch.nn as nn
from torchvision import models
from utils import make_rotation_matrix
import torch.nn.functional as F
from absl import flags
FLAGS = flags.FLAGS


class SimpleModel(nn.Module):
    def __init__(self, num_classes, use_6d=False, use_seperate_heads=False):
        super(SimpleModel, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Identity()
        self.num_classes = num_classes
        self.use_6d = use_6d
        self.use_seperate_heads = use_seperate_heads
        if use_6d:
            self.dimension_rotation = 6
        else:
            self.dimension_rotation = 9

        self.one = 1

        if use_seperate_heads:
            self.one = self.num_classes
        self.head = nn.Sequential(
                nn.Linear(512 + 4, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_classes + self.one * (self.dimension_rotation + 3))
            )

    def forward(self, image, bbox):
        # print(image.shape, bbox.shape)
        # print(image.shape)
        x = self.resnet(image)
        x = torch.cat((x, bbox), dim=1)
        x = self.head(x)
        logits, R, t = torch.split(x, [self.num_classes, self.dimension_rotation*self.one, 3*self.one], dim=1)
        return logits, R, t
    
    def rotation_6d_to_matrix(self, r):
        """
        Converts 6D rotation representation to a 3 × 3 rotation matrix using Gram-Schmidt orthogonalization.
        
        Args:
            x (torch.Tensor): Shape (batch_size, 6), where x[:, :3] is r1 and x[:, 3:] is r2.
        
        Returns:
            R (torch.Tensor): Shape (batch_size, 3, 3), valid rotation matrices.
        """
        r1 = r[:, 0:3]  # First vector
        r2 = r[:, 3:6]  # Second vector
        
        # Normalize r1
        r1 = F.normalize(r1, dim=1)

        # Make r2 orthogonal to r1
        dot_product = (r1 * r2).sum(dim=1, keepdim=True)
        r2 = r2 - dot_product * r1
        r2 = F.normalize(r2, dim=1)

        # Compute r3 as cross product of r1 and r2
        r3 = torch.cross(r1, r2, dim=1)

        # Stack into a valid 3×3 rotation matrix
        R = torch.stack((r1, r2, r3), dim=2)  # Shape: (batch_size, 3, 3)
        
        return R


    def process_output(self, outs):
        with torch.no_grad():
            logits, R, t = outs 
            cls = logits.argmax(dim=1)
            if self.use_6d:  
                R = self.rotation_6d_to_matrix(R.reshape(-1, 6))
            else:
                R = make_rotation_matrix(R.reshape(-1, 3, 3))
            t = t.reshape(-1, 3, 1)
            return cls, R, t
    
