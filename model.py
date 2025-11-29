import torch
import torch.nn as nn
import torchvision.models as models

class ProjectionHead(nn.Module):
    """α(·) - The projection head for contrastive learning"""
    def __init__(self, input_dim=2048, output_dim=32):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)

class ClassificationHead(nn.Module):
    """ψ(·) - The classifier for tumor classification"""
    # should update the input-dim when not using resnet-50
    def __init__(self, input_dim=2048, num_classes=4):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes)  # SoftMax handled by CrossEntropyLoss
        )
        
    def forward(self, x):
        return self.classifier(x)

class SSCLNet(nn.Module):
    """Complete SSCLNet model with all three blocks"""
    def __init__(self, num_classes=4, resnet_type=50):
        super(SSCLNet, self).__init__()
        
        # σ(·) - Encoder (LFG Block)
        if resnet_type == 18:
            self.encoder = models.resnet18(weights=None)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 channel for MRI
            self.feature_dim = 512
        elif resnet_type == 34:
            self.encoder = models.resnet34(weights=None)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_dim = 512
        else:  # resnet50
            self.encoder = models.resnet50(weights=None)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_dim = 2048
        
        # Remove the final classification layer
        self.encoder.fc = nn.Identity()
        
        # α(·) - Projection head (ILCL Block)
        self.projection_head = ProjectionHead(input_dim=self.feature_dim, output_dim=32)
        
        # ψ(·) - Classification head (CL Block)
        self.classifier = ClassificationHead(input_dim=self.feature_dim, num_classes=num_classes)
        
    def forward(self, x, mode="contrastive"):
        """
        Forward pass with different modes
        Args:
            x: input image
            mode: "contrastive" for pre-training, "classification" for fine-tuning
        """
        # Extract features using encoder σ(·)
        features = self.encoder(x)  # [batch_size, feature_dim]
        
        if mode == "contrastive":
            # For contrastive pre-training: return projections
            projections = self.projection_head(features)
            # Apply L2 normalization as mentioned in paper
            projections = nn.functional.normalize(projections, p=2, dim=1)
            return projections
            
        elif mode == "classification":
            # For supervised classification: return class logits
            logits = self.classifier(features)
            return logits
            
        else:
            raise ValueError("Mode must be 'contrastive' or 'classification'")
    
    def get_features(self, x):
        """Extract features without projection/classification"""
        return self.encoder(x)

# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    """NT-Xent loss from SimCLR paper"""
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, projections_a, projections_b):
        """
        Args:
            projections_a: projections from augmentation A [batch_size, projection_dim]
            projections_b: projections from augmentation B [batch_size, projection_dim]
        """
        batch_size = projections_a.shape[0]
        
        # Concatenate both augmentations
        projections = torch.cat([projections_a, projections_b], dim=0)  # [2*batch_size, projection_dim]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature  # [2*batch_size, 2*batch_size]
        
        # Create labels: positives are diagonal after concatenation
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)  # [0,1,2,...,0,1,2,...]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [2*batch_size, 2*batch_size]
        labels = labels.to(projections.device)
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(projections.device)
        labels = labels[~mask].view(2 * batch_size, -1)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        # Compute cross-entropy loss
        positives = similarity_matrix[labels.bool()].view(2 * batch_size, -1)
        negatives = similarity_matrix[~labels.bool()].view(2 * batch_size, -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long).to(projections.device)
        
        loss = nn.functional.cross_entropy(logits, labels)
        return loss