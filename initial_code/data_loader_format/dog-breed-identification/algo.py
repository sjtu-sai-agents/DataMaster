import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Dog Breed Classification Training')
    
    # Model hyperparameters
    parser.add_argument('--model_name', type=str, default='efficientnet_b3',
                        help='Model architecture name (default: efficientnet_b3)')
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='Use pretrained weights (default: True)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for dataloader (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--scheduler_t_max', type=int, default=10,
                        help='T_max for CosineAnnealingLR scheduler (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Path parameters
    parser.add_argument('--input_dir', type=str, default='./input/',
                        help='Input data directory (default: ./input/)')
    parser.add_argument('--output_dir', type=str, default='./submission/',
                        help='Output directory for submissions (default: ./submission/)')
    parser.add_argument('--working_dir', type=str, default='./working/',
                        help='Working directory for model checkpoints (default: ./working/)')
    
    # TTA parameters
    parser.add_argument('--use_tta', type=bool, default=True,
                        help='Use test-time augmentation (default: True)')
    
    return parser.parse_args()


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model and compute log loss."""
    model.eval()
    val_loss = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(target.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    preds = torch.argmax(all_probs, dim=1)
    accuracy = (preds == all_labels).float().mean().item()
    
    # Calculate log loss
    eps = 1e-15
    all_probs_clipped = torch.clamp(all_probs, eps, 1 - eps)
    log_loss = -torch.mean(torch.log(all_probs_clipped[range(len(all_labels)), all_labels]))
    
    return log_loss.item(), accuracy


def get_tta_transforms():
    """Get list of transforms for test-time augmentation."""
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.CenterCrop(320),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomResizedCrop(320, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomResizedCrop(320, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    ]
    return tta_transforms


def predict_tta(model, image_path, transforms_list, device):
    """Generate predictions using test-time augmentation."""
    image = Image.open(image_path).convert("RGB")
    all_probs = []
    
    for transform in transforms_list:
        transformed_img = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(transformed_img)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs.cpu())
    
    # Average probabilities across all augmentations
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    return avg_probs.numpy().flatten()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.working_dir, exist_ok=True)
    
    # Initialize DataLoader and get data
    data_loader = MyDataLoader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_dir=args.input_dir
    )
    train_data, test_data = data_loader.get_data()
    
    # Extract data components
    train_loader = train_data['train_loader']
    val_loader = train_data['val_loader']
    num_classes = train_data['num_classes']
    label_encoder = train_data['label_encoder']
    
    test_ids = test_data['test_ids']
    
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = timm.create_model(
        args.model_name, 
        pretrained=args.pretrained, 
        num_classes=num_classes
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.scheduler_t_max)
    
    # Training loop
    best_val_loss = float("inf")
    print("\nStarting training...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Val Log Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.working_dir, "best_model.pth"))
            print(f"Saved best model with val log loss: {val_loss:.4f}")
        
        scheduler.step()
    
    print(f"\nBest validation log loss: {best_val_loss:.4f}")
    
    # Load best model for inference
    model.load_state_dict(torch.load(os.path.join(args.working_dir, "best_model.pth")))
    model.eval()
    
    # Generate test predictions
    if args.use_tta:
        print("\nGenerating test predictions with TTA...")
        tta_transforms = get_tta_transforms()
        all_test_probs = []
        
        for test_id in tqdm(test_ids, desc="Processing test images"):
            img_path = os.path.join(args.input_dir, "test/", f"{test_id}.jpg")
            probs = predict_tta(model, img_path, tta_transforms, device)
            all_test_probs.append(probs)
        
        all_test_probs = np.array(all_test_probs)
    else:
        print("\nGenerating test predictions without TTA...")
        all_test_probs = []
        test_loader = test_data['test_loader']
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Processing test images"):
                data = data.to(device)
                output = model(data)
                probs = torch.softmax(output, dim=1)
                all_test_probs.append(probs.cpu().numpy())
        
        all_test_probs = np.concatenate(all_test_probs, axis=0)
    
    # Create submission
    sample_submission = pd.read_csv(os.path.join(args.input_dir, "sample_submission.csv"))
    submission_df = pd.DataFrame(all_test_probs, columns=label_encoder.classes_)
    submission_df.insert(0, "id", test_ids)
    submission_df = submission_df.sort_values("id")
    
    # Ensure all breed columns are present and in correct order
    expected_columns = sample_submission.columns.tolist()
    missing_cols = set(expected_columns) - set(submission_df.columns)
    if missing_cols:
        for col in missing_cols:
            submission_df[col] = 0.0
    
    # Reorder columns to match sample submission
    submission_df = submission_df[expected_columns]
    
    # Save submission file
    submission_path = os.path.join(args.output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")
    print(f"Submission shape: {submission_df.shape}")
    
    # Print validation metric
    print(f"\nFinal Validation Log Loss: {best_val_loss:.4f}")
    
    # Validate submission format
    print("\nFirst few rows of submission:")
    print(submission_df.head())
    
    print(f"\nSample submission columns: {sample_submission.shape[1]}")
    print(f"Our submission columns: {submission_df.shape[1]}")
    print(f"Column match: {set(submission_df.columns) == set(sample_submission.columns)}")
    
    # Check probabilities sum to ~1
    print(f"\nSubmission validation:")
    print(f"Mean sum of probabilities per row: {submission_df.iloc[:, 1:].sum(axis=1).mean():.6f}")
    print(f"Min probability sum: {submission_df.iloc[:, 1:].sum(axis=1).min():.6f}")
    print(f"Max probability sum: {submission_df.iloc[:, 1:].sum(axis=1).max():.6f}")


if __name__ == "__main__":
    main()