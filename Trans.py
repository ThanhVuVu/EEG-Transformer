import os
import numpy as np
import math
import random
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, (1, 51), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            # MIT-BIH shape modification: Height dimension of the kernel lowered to 2 channels
            nn.Conv2d(2, emb_size, (2, 5), stride=(1, 5)), 
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        return x

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
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
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
    # Modified parameters for MIT-BIH shape: seq_len=250, channels=2
    def __init__(self, sequence_num=250, inter=25):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)

        self.query = nn.Sequential(
            nn.Linear(2, 2),
            nn.LayerNorm(2),
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(2, 2),
            nn.LayerNorm(2),
            nn.Dropout(0.3)
        )

        self.projection = nn.Sequential(
            nn.Linear(2, 2),
            nn.LayerNorm(2),
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
    def __init__(self, emb_size=10, depth=3, n_classes=5, **kwargs):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(250), # Sequence length modified from 1000 to 250
                    channel_attention(),
                    nn.Dropout(0.5),
                )
            ),
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

class Trans():
    def __init__(self):
        super(Trans, self).__init__()
        self.batch_size = 64
        self.n_epochs = 50 # Reduced from 1000 since dataset is large (28900 train instances per epoch)
        self.img_height = 2
        self.img_width = 250
        self.channels = 1
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.9
        self.root = '/kaggle/input/datasets/shayanfazeli/heartbeat'  # Local data folder path directly provided to loader
        
        os.makedirs("./results/", exist_ok=True)
        self.log_write = open("./results/log_mitbih_training.txt", "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = ViT().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        
        # Output summary with dynamic architectural inputs
        summary(self.model, (1, 2, 250))

    def train(self):
        from getData import get_dataloaders
        
        print(f"Linking Data Pipeline to target folder: {self.root} ...")
        self.dataloader, self.test_dataloader = get_dataloaders(self.root, batch_size=self.batch_size, balance=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        bestAcc = 0
        
        print("\nFiring Training Initialized...")
        for e in range(self.n_epochs):
            self.model.train()
            train_loss_accum = 0.0
            total_train = 0
            correct_train = 0
            
            for i, (img, label) in enumerate(self.dataloader):
                img = img.cuda().type(self.Tensor)
                label = label.cuda().type(self.LongTensor)
                
                tok, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss_accum += loss.item()
                preds = torch.max(outputs, 1)[1]
                correct_train += (preds == label).sum().item()
                total_train += label.size(0)

            # Standardized validation batches to prevent large VRAM OOM error on long testing structures
            self.model.eval()
            test_loss_accum = 0.0
            total_test = 0
            correct_test = 0
            
            with torch.no_grad():
                for i, (img, label) in enumerate(self.test_dataloader):
                    img = img.cuda().type(self.Tensor)
                    label = label.cuda().type(self.LongTensor)
                    
                    tok, outputs = self.model(img)
                    loss = self.criterion_cls(outputs, label)
                    
                    test_loss_accum += loss.item()
                    preds = torch.max(outputs, 1)[1]
                    correct_test += (preds == label).sum().item()
                    total_test += label.size(0)

            train_acc = correct_train / total_train
            test_acc = correct_test / total_test
            avg_train_loss = train_loss_accum / len(self.dataloader)
            avg_test_loss = test_loss_accum / len(self.test_dataloader)
            
            print(f'Epoch: {e} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')
            self.log_write.write(f"{e}    {test_acc}\n")
            
            if test_acc > bestAcc:
                bestAcc = test_acc
                torch.save(self.model.module.state_dict(), 'model_best.pth')

        print(f'Training Deployment Concluded. High Score Testing Set Accuracy: {bestAcc:.4f}')
        self.log_write.write(f'Best Architecture Yield Accuracy: {bestAcc}\n')
        self.log_write.close()
        return bestAcc

def main():
    seed_n = 42
    print(f'Starting Monolithic Architecture Run. Base seed: {seed_n}')
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    
    trans = Trans()
    trans.train()

if __name__ == "__main__":
    main()
