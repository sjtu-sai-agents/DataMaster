import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from tqdm import tqdm


class AmbiguousDataset(Dataset):
    """Dataset for ambiguous token sequences."""
    
    def __init__(self, word, suffix, prefix, case, uidx, norm_target=None, class_target=None):
        self.word = torch.from_numpy(word)
        self.suffix = torch.from_numpy(suffix)
        self.prefix = torch.from_numpy(prefix)
        self.case = torch.from_numpy(case)
        self.uidx = torch.from_numpy(uidx)
        self.has_target = norm_target is not None
        if self.has_target:
            self.norm_target = torch.from_numpy(norm_target)
            self.class_target = torch.from_numpy(class_target)

    def __len__(self):
        return len(self.word)

    def __getitem__(self, idx):
        if self.has_target:
            return (self.word[idx], self.suffix[idx], self.prefix[idx],
                    self.case[idx], self.uidx[idx],
                    self.norm_target[idx], self.class_target[idx])
        else:
            return (self.word[idx], self.suffix[idx], self.prefix[idx],
                    self.case[idx], self.uidx[idx])


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss."""
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            n_classes = logits.size(-1)
            smooth_target = torch.ones_like(log_probs) * (self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        loss = - (smooth_target * log_probs).sum(dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class TransformerDeepCharModel(nn.Module):
    """Transformer model with character-level CNN for text normalization."""
    
    def __init__(self, word_vocab_size, suffix_vocab_size, prefix_vocab_size,
                 case_vocab_size, char_vocab_size, char_out_dim, d_model,
                 nhead, num_layers, ff_dim, dropout, max_variants, num_classes,
                 max_char_len, window_size,
                 word_emb_dim=96, suffix_emb_dim=48, prefix_emb_dim=48,
                 case_emb_dim=16, char_emb_dim=32):
        super().__init__()
        self.window_size = window_size
        self.seq_len = 2 * window_size + 1

        self.word_emb = nn.Embedding(word_vocab_size + 1, word_emb_dim, padding_idx=0)
        self.suffix_emb = nn.Embedding(suffix_vocab_size + 1, suffix_emb_dim, padding_idx=0)
        self.prefix_emb = nn.Embedding(prefix_vocab_size + 1, prefix_emb_dim, padding_idx=0)
        self.case_emb = nn.Embedding(case_vocab_size + 1, case_emb_dim, padding_idx=0)

        self.char_embed = nn.Embedding(char_vocab_size + 1, char_emb_dim, padding_idx=0)
        self.char_conv1 = nn.Conv1d(char_emb_dim, 64, kernel_size=3, padding=1)
        self.char_conv2 = nn.Conv1d(64, char_out_dim, kernel_size=3, padding=1)
        self.char_pool = nn.AdaptiveMaxPool1d(1)
        self.char_act = nn.GELU()
        self.char_dropout = nn.Dropout(dropout)

        total_emb_dim = word_emb_dim + suffix_emb_dim + prefix_emb_dim + case_emb_dim + char_out_dim
        self.proj = nn.Linear(total_emb_dim, d_model)

        self.pos_embed = nn.Embedding(self.seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        self.center_idx = window_size
        combined_dim = d_model * 3
        self.norm_head = nn.Linear(combined_dim, max_variants)
        self.class_head = nn.Linear(combined_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, word_seq, suffix_seq, prefix_seq, case_seq, uidx_seq, char_matrix):
        word_seq = word_seq.long()
        suffix_seq = suffix_seq.long()
        prefix_seq = prefix_seq.long()
        case_seq = case_seq.long()
        uidx_seq = uidx_seq.long()

        batch_size, seq_len = word_seq.size()
        device = word_seq.device

        word_emb = self.word_emb(word_seq)
        suffix_emb = self.suffix_emb(suffix_seq)
        prefix_emb = self.prefix_emb(prefix_seq)
        case_emb = self.case_emb(case_seq)

        # Character features
        flat_uidx = uidx_seq.view(-1)
        char_indices = char_matrix[flat_uidx]
        char_emb = self.char_embed(char_indices)
        char_emb = char_emb.permute(0, 2, 1)
        char_feat = self.char_act(self.char_conv1(char_emb))
        char_feat = self.char_act(self.char_conv2(char_feat))
        char_feat = self.char_pool(char_feat).squeeze(-1)
        char_feat = self.char_dropout(char_feat)
        char_feat = char_feat.view(batch_size, seq_len, -1)

        combined = torch.cat([word_emb, suffix_emb, prefix_emb, case_emb, char_feat], dim=2)
        combined = self.proj(combined)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)
        x = combined + pos_emb
        x = self.dropout(x)

        x = self.transformer(x)

        center = x[:, self.center_idx, :]
        avg_pool = torch.mean(x, dim=1)
        max_pool = torch.max(x, dim=1)[0]
        agg = torch.cat([center, avg_pool, max_pool], dim=1)
        agg = self.dropout(agg)

        norm_logits = self.norm_head(agg)
        class_logits = self.class_head(agg)
        return norm_logits, class_logits


def train_epoch(model, loader, optimizer, scheduler,
                norm_criterion, class_criterion, device, char_matrix, class_loss_weight):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct_norm = 0
    total = 0
    for batch in tqdm(loader, desc='Training', leave=False):
        word, suffix, prefix, case, uidx, norm_target, class_target = [x.to(device) for x in batch]
        optimizer.zero_grad()
        norm_logits, class_logits = model(word, suffix, prefix, case, uidx, char_matrix)
        loss_norm = norm_criterion(norm_logits, norm_target)
        loss_class = class_criterion(class_logits, class_target)
        loss = loss_norm + class_loss_weight * loss_class
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * word.size(0)
        pred_norm = norm_logits.argmax(dim=1)
        correct_norm += (pred_norm == norm_target).sum().item()
        total += word.size(0)

    avg_loss = total_loss / total if total else 0
    accuracy = correct_norm / total if total else 0
    return avg_loss, accuracy


def evaluate(model, loader, device, char_matrix):
    """Evaluate model on validation set."""
    model.eval()
    correct_norm = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            word, suffix, prefix, case, uidx, norm_target, class_target = [x.to(device) for x in batch]
            norm_logits, _ = model(word, suffix, prefix, case, uidx, char_matrix)
            pred_norm = norm_logits.argmax(dim=1)
            correct_norm += (pred_norm == norm_target).sum().item()
            total += word.size(0)
    accuracy = correct_norm / total if total else 0
    return accuracy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Transformer model for text normalization')
    
    # Model hyperparameters
    parser.add_argument('--word_emb_dim', type=int, default=96, help='Word embedding dimension')
    parser.add_argument('--suffix_emb_dim', type=int, default=48, help='Suffix embedding dimension')
    parser.add_argument('--prefix_emb_dim', type=int, default=48, help='Prefix embedding dimension')
    parser.add_argument('--case_emb_dim', type=int, default=16, help='Case embedding dimension')
    parser.add_argument('--char_emb_dim', type=int, default=32, help='Character embedding dimension')
    parser.add_argument('--char_out_dim', type=int, default=128, help='Character CNN output dimension')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--ff_dim', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--class_loss_weight', type=float, default=0.5, help='Class loss weight')
    
    # Paths
    parser.add_argument('--output_dir', type=str, default='./submission', help='Output directory')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data using MyDataLoader
    print('Loading data...')
    data_loader = MyDataLoader()
    train_data, test_data = data_loader.get_data()
    
    # Extract training data
    X_word = train_data['X_word']
    X_suffix = train_data['X_suffix']
    X_prefix = train_data['X_prefix']
    X_case = train_data['X_case']
    X_uidx = train_data['X_uidx']
    y_norm = train_data['y_norm']
    y_class = train_data['y_class']
    sentence_ids = train_data['sentence_ids']
    val_sentence_ids = train_data['val_sentence_ids']
    
    # Extract metadata
    vocab_size = train_data['vocab_size']
    suffix_vocab_size = train_data['suffix_vocab_size']
    prefix_vocab_size = train_data['prefix_vocab_size']
    max_variants = train_data['max_variants']
    window_size = train_data['window_size']
    max_char_len = train_data['max_char_len']
    num_classes = train_data['num_classes']
    char_vocab_size = train_data['char_vocab_size']
    char_matrix = train_data['char_matrix']
    
    # Move character matrix to device
    char_matrix_tensor = torch.from_numpy(char_matrix).long().to(device)
    
    # Prepare data for CV
    X = {
        'word': X_word,
        'suffix': X_suffix,
        'prefix': X_prefix,
        'case': X_case,
        'uidx': X_uidx,
        'norm': y_norm,
        'class': y_class
    }
    groups = sentence_ids
    
    # Cross-validation setup
    if val_sentence_ids is not None:
        # Use fixed validation set from val.csv
        print('Using fixed validation set from val.csv')
        val_mask = np.isin(sentence_ids, list(val_sentence_ids))
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        folds = [(train_idx, val_idx)]
        n_folds = 1
    else:
        # Use GroupKFold
        kf = GroupKFold(n_splits=args.n_folds)
        folds = list(kf.split(X['word'], groups=groups))
        n_folds = args.n_folds
    
    fold_val_accs = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f'\n--- Fold {fold + 1}/{n_folds} ---')
        
        # Create datasets
        train_dataset = AmbiguousDataset(
            X['word'][train_idx], X['suffix'][train_idx], X['prefix'][train_idx],
            X['case'][train_idx], X['uidx'][train_idx],
            X['norm'][train_idx], X['class'][train_idx]
        )
        val_dataset = AmbiguousDataset(
            X['word'][val_idx], X['suffix'][val_idx], X['prefix'][val_idx],
            X['case'][val_idx], X['uidx'][val_idx],
            X['norm'][val_idx], X['class'][val_idx]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        # Initialize model
        model = TransformerDeepCharModel(
            word_vocab_size=vocab_size,
            suffix_vocab_size=suffix_vocab_size,
            prefix_vocab_size=prefix_vocab_size,
            case_vocab_size=4,
            char_vocab_size=char_vocab_size,
            char_out_dim=args.char_out_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            max_variants=max_variants,
            num_classes=num_classes,
            max_char_len=max_char_len,
            window_size=window_size,
            word_emb_dim=args.word_emb_dim,
            suffix_emb_dim=args.suffix_emb_dim,
            prefix_emb_dim=args.prefix_emb_dim,
            case_emb_dim=args.case_emb_dim,
            char_emb_dim=args.char_emb_dim
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
        )
        norm_criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        class_criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_state = None
        
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler,
                norm_criterion, class_criterion, device, char_matrix_tensor,
                args.class_loss_weight
            )
            val_acc = evaluate(model, val_loader, device, char_matrix_tensor)
            elapsed = time.time() - start
            print(f'Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.0f}s')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        print(f'Fold {fold + 1} best validation accuracy: {best_val_acc:.4f}')
        fold_val_accs.append(best_val_acc)
        
        # Load best model
        model.load_state_dict(best_state)
        model.to(device)
        models.append(model)
    
    # Print CV results
    print('\nCross-validation results:')
    for fold, acc in enumerate(fold_val_accs):
        print(f'Fold {fold + 1}: {acc:.4f}')
    print(f'Average: {np.mean(fold_val_accs):.4f}')
    
    # Inference on test set
    print('\nPreparing test data...')
    test_ambig = test_data['test_ambig']
    test_nonambig = test_data['test_nonambig']
    test_word_seq = test_data['test_word_seq']
    test_suffix_seq = test_data['test_suffix_seq']
    test_prefix_seq = test_data['test_prefix_seq']
    test_case_seq = test_data['test_case_seq']
    test_uidx_seq = test_data['test_uidx_seq']
    token_map = test_data['token_map']
    default_after_map = test_data['default_after_map']
    
    test_ambig_dataset = AmbiguousDataset(
        test_word_seq, test_suffix_seq, test_prefix_seq,
        test_case_seq, test_uidx_seq
    )
    test_ambig_loader = DataLoader(
        test_ambig_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Accumulate probabilities from all folds
    all_probs = np.zeros((len(test_ambig), max_variants), dtype=np.float32)
    
    for model in models:
        model.eval()
        probs_fold = []
        with torch.no_grad():
            for batch in tqdm(test_ambig_loader, desc='Fold inference', leave=False):
                word, suffix, prefix, case, uidx = [x.to(device) for x in batch]
                logits, _ = model(word, suffix, prefix, case, uidx, char_matrix_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_fold.append(probs)
        probs_fold = np.vstack(probs_fold)
        all_probs += probs_fold
    
    all_probs /= len(models)
    
    # Map to final after strings
    ambig_predictions = []
    for i, (_, row) in enumerate(test_ambig.iterrows()):
        before = row['before']
        cand = token_map.get(before, [])
        if not cand:
            pred_after = before
        else:
            probs_i = all_probs[i, :len(cand)]
            pred_idx = np.argmax(probs_i)
            pred_after = cand[pred_idx]
        ambig_predictions.append(pred_after)
    
    test_ambig = test_ambig.copy()
    test_ambig['after'] = ambig_predictions
    
    # For non-ambiguous tokens
    test_nonambig = test_nonambig.copy()
    test_nonambig['after'] = test_nonambig['before'].map(default_after_map).fillna(test_nonambig['before'])
    
    # Combine back
    test_full = pd.concat([test_ambig, test_nonambig], axis=0).sort_index()
    
    # Create submission
    test_full['id'] = test_full['sentence_id'].astype(str) + '_' + test_full['token_id'].astype(str)
    submission = test_full[['id', 'after']]
    
    # Ensure output directory
    os.makedirs(args.output_dir, exist_ok=True)
    submission.to_csv(os.path.join(args.output_dir, 'submission.csv'), index=False, quoting=1)
    print(f'Submission saved to {args.output_dir}/submission.csv')
    print('All done.')


if __name__ == "__main__":
    main()