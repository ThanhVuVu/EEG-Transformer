import os
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchsummary import summary

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.projection = nn.Sequential(
            # Spatial dimensions: 1 Depth, 12 Channels, 500 Sequence Length
            # 500 - 51 + 1 = 450 output length
            nn.Conv2d(1, 2, (1, 51), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            # Compressing the 12 semantic spatial channels into standard patch lengths
            nn.Conv2d(2, emb_size, (12, 5), stride=(12, 5)), 
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        # Residual/skip connection: x + fn(x)
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=5, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                # Attention sublayer with residual add
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                # Feed-forward sublayer with residual add
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
    def forward(self, x):
        out = self.clshead(x)
        return x, out

class channel_attention(nn.Module):
    def __init__(self, sequence_num=500, inter=10):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)

        # Re-scaled for exact 12 spatial clinical channels
        self.query = nn.Sequential(
            nn.Linear(12, 12),
            nn.LayerNorm(12),
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(12, 12),
            nn.LayerNorm(12),
            nn.Dropout(0.3)
        )

        self.projection = nn.Sequential(
            nn.Linear(12, 12),
            nn.LayerNorm(12),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)
        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling
        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out

class ViT(nn.Sequential):
    def __init__(self, emb_size=10, depth=3, n_classes=4, **kwargs):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    # Initial channel-attention block with residual add
                    nn.LayerNorm(500), # Norm across exact 500 subsequence timeline
                    channel_attention(sequence_num=500, inter=10),
                    nn.Dropout(0.5),
                )
            ),
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes) # Bound to the 4 diagnostic nodes
        )

class Trans():
    def __init__(self):
        super(Trans, self).__init__()
        self.batch_size = 64
        self.n_epochs = 50
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.9
        self.root = '/kaggle/input/datasets/erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database/WFDB_ChapmanShaoxing'
        
        os.makedirs("./results/", exist_ok=True)
        self.log_write = open("./results/log_chapman_metrics.txt", "w")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)
        self.model = ViT(n_classes=4).to(self.device)

        # Smoke test: run a few forward/backward/optimizer steps on a small fraction
        # of train/val to verify the pipeline. After smoke, we restore weights so
        # full training is unaffected.
        self.smoke_fraction = 0.05
        self.smoke_max_batches = 10
        
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        
        try:
            summary(self.model, (1, 12, 500), device=self.device.type)
        except Exception as e:
            print("Summary generation skipped structure matrix:", e)

    def compute_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        # Macro averaging utilized to weigh minor arrhythmia boundaries identically
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return acc, prec, rec, f1

    def train(self):
        from getData import get_dataloaders
        
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(self.root, batch_size=self.batch_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # -------------------------
        # Non-destructive smoke run
        # -------------------------
        if self.smoke_fraction and self.smoke_fraction > 0:
            # Save initial weights (for DataParallel this includes the 'module.' prefix).
            initial_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            smoke_train_steps = int(len(self.train_loader) * self.smoke_fraction) if len(self.train_loader) > 0 else 0
            smoke_train_steps = min(len(self.train_loader), self.smoke_max_batches, max(1, smoke_train_steps))

            smoke_val_steps = int(len(self.val_loader) * self.smoke_fraction) if len(self.val_loader) > 0 else 0
            smoke_val_steps = min(len(self.val_loader), self.smoke_max_batches, max(1, smoke_val_steps))

            if smoke_train_steps == 0 or smoke_val_steps == 0:
                print("[SMOKE] Skipping smoke run because one of the dataloaders is empty.")
            else:
                print(f"\n[SMOKE] Running for up to {smoke_train_steps} train batches and {smoke_val_steps} val batches...")
                self.log_write.write(f"[SMOKE] Running for up to {smoke_train_steps} train batches and {smoke_val_steps} val batches...\n")

                self.model.train()
                smoke_train_loss_accum = 0.0
                y_true_train, y_pred_train = [], []

                for i, (img, label) in enumerate(self.train_loader):
                    if i >= smoke_train_steps:
                        break

                    # Dataset contract: img is (B, 1, 12, 500), label is (B,) with values in {0,1,2,3}.
                    img = img.to(self.device)
                    label = label.to(self.device)

                    tok, outputs = self.model(img)
                    # Model contract: outputs are raw logits with shape (B, 4); tok is the transformer tokens.
                    loss = self.criterion_cls(outputs, label)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    smoke_train_loss_accum += loss.item()
                    preds = torch.max(outputs, 1)[1]
                    y_true_train.extend(label.cpu().numpy())
                    y_pred_train.extend(preds.cpu().numpy())

                self.model.eval()
                smoke_val_loss_accum = 0.0
                y_true_val, y_pred_val = [], []

                with torch.no_grad():
                    for i, (img, label) in enumerate(self.val_loader):
                        if i >= smoke_val_steps:
                            break

                        img = img.to(self.device)
                        label = label.to(self.device)

                        tok, outputs = self.model(img)
                        loss = self.criterion_cls(outputs, label)

                        smoke_val_loss_accum += loss.item()
                        preds = torch.max(outputs, 1)[1]
                        y_true_val.extend(label.cpu().numpy())
                        y_pred_val.extend(preds.cpu().numpy())

                tr_acc, tr_prec, tr_rec, tr_f1 = self.compute_metrics(y_true_train, y_pred_train)
                vl_acc, vl_prec, vl_rec, vl_f1 = self.compute_metrics(y_true_val, y_pred_val)

                avg_smoke_train_loss = smoke_train_loss_accum / max(1, smoke_train_steps)
                avg_smoke_val_loss = smoke_val_loss_accum / max(1, smoke_val_steps)

                smoke_str = (f"[SMOKE RESULTS] "
                             f"Train Loss: {avg_smoke_train_loss:.4f} | Train F1: {tr_f1:.4f} "
                             f"| Val Loss: {avg_smoke_val_loss:.4f} | Val F1: {vl_f1:.4f}")
                print(smoke_str)
                self.log_write.write(smoke_str + "\n")

                # Restore original weights so the later training is unaffected.
                self.model.load_state_dict(initial_state)
                # Reset optimizer state for a clean start.
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        best_f1 = 0
        
        for e in range(self.n_epochs):
            self.model.train()
            train_loss_accum = 0.0
            y_true_train, y_pred_train = [], []
            
            for i, (img, label) in enumerate(self.train_loader):
                img = img.to(self.device)
                label = label.to(self.device)
                
                tok, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss_accum += loss.item()
                preds = torch.max(outputs, 1)[1]
                
                y_true_train.extend(label.cpu().numpy())
                y_pred_train.extend(preds.cpu().numpy())

            self.model.eval()
            val_loss_accum = 0.0
            y_true_val, y_pred_val = [], []
            
            with torch.no_grad():
                for i, (img, label) in enumerate(self.val_loader):
                    img = img.to(self.device)
                    label = label.to(self.device)
                    
                    tok, outputs = self.model(img)
                    loss = self.criterion_cls(outputs, label)
                    
                    val_loss_accum += loss.item()
                    preds = torch.max(outputs, 1)[1]
                    
                    y_true_val.extend(label.cpu().numpy())
                    y_pred_val.extend(preds.cpu().numpy())

            # Evaluate Metrics
            tr_acc, tr_prec, tr_rec, tr_f1 = self.compute_metrics(y_true_train, y_pred_train)
            vl_acc, vl_prec, vl_rec, vl_f1 = self.compute_metrics(y_true_val, y_pred_val)
            
            avg_train_loss = train_loss_accum / len(self.train_loader)
            avg_val_loss = val_loss_accum / len(self.val_loader)
            
            out_str = (f"Epoch: {e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
                       f"| Val Acc: {vl_acc:.4f} | Prec: {vl_prec:.4f} | Rec: {vl_rec:.4f} | F1: {vl_f1:.4f}")
            print(out_str)
            self.log_write.write(out_str + "\n")
            
            if vl_f1 > best_f1:
                best_f1 = vl_f1
                state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                torch.save(state, 'model_chapman_best.pth')

        # Run Final Test set validation against highest Val boundaries
        print("\nDeploying Independent Testing Pipeline on Held-out Set...")
        self.model.load_state_dict(torch.load('model_chapman_best.pth'))
        self.model.eval()
        y_true_ts, y_pred_ts = [], []
        
        with torch.no_grad():
            for i, (img, label) in enumerate(self.test_loader):
                img = img.to(self.device)
                label = label.to(self.device)
                tok, outputs = self.model(img)
                preds = torch.max(outputs, 1)[1]
                y_true_ts.extend(label.cpu().numpy())
                y_pred_ts.extend(preds.cpu().numpy())
                
        ts_acc, ts_prec, ts_rec, ts_f1 = self.compute_metrics(y_true_ts, y_pred_ts)
        test_str = (f"\n[FINAL TEST SET] Accuracy: {ts_acc:.4f} | Precision: {ts_prec:.4f} "
                    f"| Recall: {ts_rec:.4f} | F1-Score: {ts_f1:.4f}")
        print(test_str)
        self.log_write.write(test_str + "\n")
        self.log_write.close()

def main():
    seed_n = 42
    print(f'Initializing 12-Lead Space Matrix... Random seed bound to {seed_n}')
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
    
    trans = Trans()
    trans.train()

if __name__ == "__main__":
    main()
