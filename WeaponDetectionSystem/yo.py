import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
import os
from PIL import Image
import numpy as np

# Define your dataset class
class WeaponDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # Load all image and annotation files
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annots = list(sorted(os.listdir(os.path.join(root, "labels"))))
        
    def __getitem__(self, idx):
        # Load image and annotations
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "labels", self.annots[idx])
        
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)
        
        # Parse your annotation file and convert to boxes, labels
        boxes, labels = self.parse_annotation(annot_path)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    def __len__(self):
        return len(self.imgs)
    
    def parse_annotation(self, annot_path):
        # Implement your annotation parsing logic here
        # Return boxes (tensor) and labels (tensor)
        pass

# Load a pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the classifier with a new one for your number of classes
num_classes = 9  # 8 weapons + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Define training and validation datasets
dataset_train = WeaponDataset("train")
dataset_val = WeaponDataset("val")

# Define data loaders
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=lambda x: tuple(zip(*x)))

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=lambda x: tuple(zip(*x)))

# Training parameters
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader_train:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    lr_scheduler.step()
    
    # Validation
    model.eval()
    # Add your validation metrics here