# ==============================================================================
# File: scripts/train_classifier.py
# Purpose: Train Stage 2 Classifier on ground-truth crops (crops/train, crops/val)
# ==============================================================================

import argparse
import json
import os
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import torch

# Switch PyTorch IPC sharing strategy from shared memory (/dev/shm) to disk (/tmp)
torch.multiprocessing.set_sharing_strategy('file_system')

class TaxonomicCropDataset(Dataset):
    """
    Dataset loader for ground-truth crops organized into folders:
      crops/train/<class_name>/crop_xxx.png
      crops/val/<class_name>/crop_xxx.png
    """
    def __init__(self, root_dir: Path, taxonomy_config: dict, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        # Load class targets from taxonomy JSON configuration
        classes = taxonomy_config["classes"]
        for class_id, info in classes.items():
            folder_name = info["name"]
            folder_path = self.root_dir / folder_name
            
            if not folder_path.exists():
                continue

            genus_id = info["genus_id"]
            species_id = info["species_id"]

            for img_path in folder_path.glob("*.png"):
                self.samples.append((
                    str(img_path),
                    torch.tensor(genus_id, dtype=torch.long),
                    torch.tensor(species_id, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, genus_target, species_target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, genus_target, species_target


class HierarchicalTaxonomicClassifier(nn.Module):
    """
    Two-head PyTorch classifier extending a timm vision backbone.
    Head 1: Genus branch
    Head 2: Species branch
    """
    def __init__(self, backbone_name: str = "convnext_tiny", num_genera: int = 1, num_species: int = 2, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features

        self.genus_head = nn.Linear(in_features, num_genera)
        self.species_head = nn.Linear(in_features, num_species)

    def forward(self, x):
        feats = self.backbone(x)
        genus_logits = self.genus_head(feats)
        species_logits = self.species_head(feats)
        return genus_logits, species_logits

class MaskedHierarchicalLoss(nn.Module):
    def __init__(self, ignore_species_idx: int = -100):
        super().__init__()
        self.ignore_species_idx = ignore_species_idx
        self.genus_criterion = nn.CrossEntropyLoss()
        # Set reduction='sum' so we can manually divide by valid target counts
        self.species_criterion = nn.CrossEntropyLoss(ignore_index=ignore_species_idx, reduction='sum')

    def forward(self, genus_logits, species_logits, genus_targets, species_targets):
        # 1. Genus loss (always computed)
        loss_genus = self.genus_criterion(genus_logits, genus_targets)

        # 2. Species loss (guarded against zero valid targets)
        valid_mask = (species_targets != self.ignore_species_idx)
        n_valid = valid_mask.sum().item()

        if n_valid > 0:
            loss_species = self.species_criterion(species_logits, species_targets) / n_valid
        else:
            # If batch contains only 'cancer_sp' (-100), species loss is zero
            loss_species = torch.tensor(0.0, device=genus_logits.device)

        return loss_genus + loss_species


def get_transforms(img_size: int = 224, is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, genus_targets, species_targets in loader:
        images = images.to(device)
        genus_targets = genus_targets.to(device)
        species_targets = species_targets.to(device)

        optimizer.zero_grad()
        genus_logits, species_logits = model(images)
        loss = criterion(genus_logits, species_logits, genus_targets, species_targets)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_species = 0
    total_species = 0

    for images, genus_targets, species_targets in loader:
        images = images.to(device)
        genus_targets = genus_targets.to(device)
        species_targets = species_targets.to(device)

        genus_logits, species_logits = model(images)
        loss = criterion(genus_logits, species_logits, genus_targets, species_targets)
        running_loss += loss.item() * images.size(0)

        # Evaluate accuracy on fine-grained species labels only (skip -100 masked targets)
        valid_mask = species_targets != -100
        if valid_mask.sum() > 0:
            preds = species_logits[valid_mask].argmax(dim=1)
            correct_species += (preds == species_targets[valid_mask]).sum().item()
            total_species += valid_mask.sum().item()

    acc = (correct_species / total_species) if total_species > 0 else 0.0
    return running_loss / len(loader.dataset), acc


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 Hierarchical Classifier")
    parser.add_argument("--crop_dir", default="crops", help="Root folder containing train/ and val/ crops")
    parser.add_argument("--taxonomy_json", default="config/taxonomy_config.json", help="Path to taxonomy config")
    parser.add_argument("--backbone", default="convnext_tiny", help="Timm backbone architecture")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_weights", default="output/stage2_classifier.pt")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.taxonomy_json, "r") as f:
        taxonomy_config = json.load(f)

    # Prepare datasets
    train_dataset = TaxonomicCropDataset(Path(args.crop_dir) / "train", taxonomy_config, get_transforms(224, is_train=True))
    val_dataset = TaxonomicCropDataset(Path(args.crop_dir) / "val", taxonomy_config, get_transforms(224, is_train=False))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,  # <-- Set to 0 to run data loading on the main process
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,  # <-- Set to 0
        pin_memory=True
    )

    # Initialize model
    species_id_map = taxonomy_config.get("species_id_to_name", {"0": "borealis", "1": "irroratus"})
    model = HierarchicalTaxonomicClassifier(
        backbone_name=args.backbone,
        num_genera=1,
        num_species=len(species_id_map)
    ).to(device)

    criterion = MaskedHierarchicalLoss(ignore_species_idx=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    print(f"Training Stage 2 ({args.backbone}) on {len(train_dataset)} crops | Val: {len(val_dataset)} crops")

    best_val_acc = 0.0
    Path(args.out_weights).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Species Acc: {val_acc*100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out_weights)
            print(f"  --> Saved new best model to {args.out_weights}")

    print("Stage 2 Training Complete.")


if __name__ == "__main__":
    main()