"""
model.py - ResNet-9 Deep Learning Architecture for Plant Disease Classification
Extracted and modularized from:
- Copy of plant-disease-classification-resnet-99-2.ipynb
- plant_desease_model_load.ipynb

Achieves 99.2% validation accuracy on the 38-class plant disease dataset.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# The 38 classes from the New Plant Diseases Dataset (Augmented)
# Covering staple crops including Maize/Corn, Potato, Tomato, Apple, etc.
PLANT_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Additional regional African staple crop diseases (DOCX context)
NIGERIAN_EXTENDED_CLASSES = [
    'Cassava___Cassava_Mosaic_Disease',
    'Cassava___Cassava_Brown_Streak_Disease',
    'Cassava___Bacterial_Blight',
    'Cassava___healthy',
    'Corn_(maize)___Fall_Armyworm_lesion'
]


def get_default_device():
    """Pick GPU if available, else CPU (standard for Raspberry Pi companion computer)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device."""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device."""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches."""
        return len(self.dl)


def accuracy(outputs, labels):
    """Calculate multiclass classification accuracy."""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    """Base class for image classification with training and validation step metrics."""

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,
                result.get("lrs", [-1])[-1],
                result.get("train_loss", 0.0),
                result.get("val_loss", 0.0),
                result.get("val_accuracy", 0.0),
            )
        )


class SimpleResidualBlock(nn.Module):
    """Standard residual block with 3x3 convolutions and skip connection."""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x


def ConvBlock(in_channels, out_channels, pool=False):
    """Convolution block with Batch Normalization and optional 4x Max Pooling."""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(ImageClassificationBase):
    """
    ResNet-9 Convolutional Neural Network Architecture for Plant Disease Classification.
    Input image dimension: 3 x 256 x 256
    Model parameter size: ~6.5 Million parameters (~25 MB weights), ideal for embedded ARM edge deployment.
    """

    def __init__(self, in_channels=3, num_diseases=38):
        super().__init__()

        # Conv1: 3 -> 64 (256x256)
        self.conv1 = ConvBlock(in_channels, 64)
        # Conv2: 64 -> 128 with MaxPool4 (64x64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        # Res1: 128 -> 128 -> 128
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        # Conv3: 128 -> 256 with MaxPool4 (16x16)
        self.conv3 = ConvBlock(128, 256, pool=True)
        # Conv4: 256 -> 512 with MaxPool4 (4x4)
        self.conv4 = ConvBlock(256, 512, pool=True)
        # Res2: 512 -> 512 -> 512
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        # Classifier: MaxPool4 -> Flatten -> Linear(512, num_diseases)
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


def load_trained_resnet(model_path=None, num_classes=len(PLANT_CLASSES), device=None):
    """
    Factory function to initialize and safely load ResNet9 weights.
    Fixes the critical bug in plant_desease_model_load.ipynb where weights were overwritten.
    """
    if device is None:
        device = get_default_device()

    model = ResNet9(in_channels=3, num_diseases=num_classes)
    model.to(device)

    if model_path and os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            elif isinstance(state, nn.Module):
                model = state
                model.to(device)
            print(f"[ModelLoader] Loaded weights successfully from: {model_path}")
        except Exception as e:
            print(f"[ModelLoader] Warning: Could not load weights from {model_path} ({e}). Using initialized model.")
    else:
        if model_path:
            print(f"[ModelLoader] Model path '{model_path}' not found. Using initialized ResNet-9.")

    model.eval()
    try:
        dummy_in = torch.zeros(1, 3, 256, 256, device=device)
        traced = torch.jit.trace(model, dummy_in)
        frozen = torch.jit.freeze(traced)
        optimized = torch.jit.optimize_for_inference(frozen)
        return optimized
    except Exception:
        return model
