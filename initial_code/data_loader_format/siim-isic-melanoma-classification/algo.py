import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Melanoma Classification Training')
    
    # Model hyperparameters
    parser.add_argument('--meta_dim', type=int, default=3,
                        help='Dimension of metadata features')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for resizing')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_patience', type=int, default=2,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor for learning rate scheduler')
    
    # Path parameters
    parser.add_argument('--input_path', type=str, default='./input',
                        help='Path to input data directory')
    parser.add_argument('--working_path', type=str, default='./working',
                        help='Path to working directory for saving models')
    parser.add_argument('--submission_path', type=str, default='./submission',
                        help='Path to save submission file')
    
    return parser.parse_args()


class MelanomaModel(nn.Module):
    """Melanoma classification model combining image and metadata features."""
    
    def __init__(self, meta_dim=3):
        super(MelanomaModel, self).__init__()
        # Image branch (ResNet50)
        self.img_model = models.resnet50(pretrained=True)
        self.img_model.fc = nn.Identity()
        img_features = 2048

        # Metadata branch
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        meta_features = 32

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(img_features + meta_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, meta):
        # Image features
        img_features = self.img_model(img)

        # Metadata features
        meta_features = self.meta_fc(meta)

        # Concatenate and classify
        combined = torch.cat([img_features, meta_features], dim=1)
        output = self.classifier(combined)
        return output.squeeze()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.working_path, exist_ok=True)
    os.makedirs(args.submission_path, exist_ok=True)
    
    # Initialize data loader and get data
    data_loader = MyDataLoader(
        input_path=args.input_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    train_data, test_data = data_loader.get_data()
    
    train_loader = train_data['train_loader']
    val_loader = train_data['val_loader']
    val_df = train_data['val_df']
    test_loader = test_data['test_loader']
    test_df = test_data['test_df']
    
    # Initialize model
    model = MelanomaModel(meta_dim=args.meta_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.scheduler_patience, factor=args.scheduler_factor
    )
    
    # Training loop
    best_val_auc = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for images, metadata, targets in train_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(images, metadata)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, metadata, targets in val_loader:
                images = images.to(device)
                metadata = metadata.to(device)
                targets = targets.float().to(device)

                outputs = model(images, metadata)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        train_auc = roc_auc_score(train_targets, train_preds)
        val_auc = roc_auc_score(val_targets, val_preds)
        avg_train_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Update scheduler
        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.working_path, "best_model.pth"))

    print(f"\nBest Validation AUC: {best_val_auc:.4f}")
    
    # Load best model and predict on test set
    model.load_state_dict(torch.load(os.path.join(args.working_path, "best_model.pth")))
    model.eval()

    test_preds = []

    with torch.no_grad():
        for images, metadata in test_loader:
            images = images.to(device)
            metadata = metadata.to(device)

            outputs = model(images, metadata)
            test_preds.extend(outputs.cpu().numpy())

    # Create submission file
    submission_df = pd.DataFrame({
        "image_name": test_df["image_name"],
        "target": test_preds
    })
    
    submission_file = os.path.join(args.submission_path, "submission.csv")
    submission_df.to_csv(submission_file, index=False)

    print(f"\nSubmission saved to {submission_file}")
    print(f"Submission shape: {submission_df.shape}")
    print(f"Target range: [{submission_df['target'].min():.3f}, {submission_df['target'].max():.3f}]")

    # Save validation predictions for reference
    val_df_copy = val_df.copy()
    val_df_copy["prediction"] = val_preds
    val_df_copy[["image_name", "target", "prediction"]].to_csv(
        os.path.join(args.working_path, "validation_predictions.csv"), index=False
    )
    print(f"\nValidation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()