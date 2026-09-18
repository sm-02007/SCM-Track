import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms


class PieceClassifier:
    def __init__(self, model_path="data/model.pth", device="cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)

        self.classes = checkpoint["classes"]
        self.img_size = checkpoint.get("img_size", 128)

        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features, len(self.classes)
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, crop_bgr: np.ndarray) -> str:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            pred_idx = output.argmax(1).item()

        return self.classes[pred_idx]