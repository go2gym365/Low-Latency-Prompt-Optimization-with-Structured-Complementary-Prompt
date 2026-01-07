#!/usr/bin/env python3
import json
import os
os.environ['HF_HOME'] = '/nas/user77/workspace/models'
os.environ['HF_CACHE'] = '/nas/user77/workspace/models'
os.environ['TRANSFORMERS_CACHE'] = '/nas/user77/workspace/models'
import random
import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast, DataCollatorWithPadding, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm




# Set seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        logpt = nn.functional.log_softmax(inputs, dim=-1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            alpha_factor = self.alpha[targets].to(inputs.device)
            logpt = logpt * alpha_factor
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()

class MultiTaskDataset(Dataset):
    def __init__(self, data, tokenizer, max_length= 512):
        self.texts = [item['prompt'] for item in data]
        self.labels = [item['label'] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        for task, lbl in self.labels[idx].items():
            item[f"label_{task}"] = torch.tensor(lbl, dtype=torch.long)
        return item

class MultiTaskBERT(nn.Module):
    def __init__(self, model_name: str, num_classes: dict, dropout_rate: float, gamma: float):
        super().__init__()


        self.bert = AutoModel.from_pretrained(
            model_name, 
        )

        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        # classifiers per task
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(hidden_size, n_classes)
            for task, n_classes in num_classes.items()
        })
        # replace cross-entropy with focal loss
        self.loss_fns = {task: FocalLoss(gamma=gamma) for task in num_classes}

    def forward(self, input_ids, attention_mask, **labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # ModernBERT는 pooler_output이 없으므로 last_hidden_state의 [CLS] 토큰 사용
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = self.dropout(outputs.pooler_output)
        else:
            # ModernBERT나 다른 모델: [CLS] 토큰 (첫 번째 토큰) 사용
            pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = {task: clf(pooled) for task, clf in self.classifiers.items()}
        loss = None
        # compute focal loss if labels provided
        if all(f'label_{task}' in labels for task in self.classifiers):
            loss = sum(
                self.loss_fns[task](logits[task], labels[f'label_{task}'])
                for task in self.classifiers
            )
        return {'loss': loss, **logits}

# evaluation function

def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_steps = 0.0, 0
    all_preds, all_labels = defaultdict(list), defaultdict(list)
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                **{k: batch[k] for k in batch if k.startswith('label_')}
            )
            if outputs['loss'] is not None:
                total_loss += outputs['loss'].item()
                total_steps += 1
            for task in model.classifiers:
                preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                lbls = batch[f'label_{task}'].cpu().numpy()
                all_preds[task].extend(preds)
                all_labels[task].extend(lbls)
    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    metrics, f1_scores = {}, []
    for task in all_preds:
        acc = accuracy_score(all_labels[task], all_preds[task])
        prec = precision_score(all_labels[task], all_preds[task], average='macro', zero_division=0)
        rec = recall_score(all_labels[task], all_preds[task], average='macro', zero_division=0)
        f1 = f1_score(all_labels[task], all_preds[task], average='macro', zero_division=0)
        metrics[task] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
        f1_scores.append(f1)
    metrics['avg_f1'] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    metrics['val_loss'] = avg_loss
    model.train()
    return metrics

# main training script with batch accumulation

def main():
    parser = argparse.ArgumentParser(description='Multi-task BERT with Focal Loss and Batch Accumulation')
    parser.add_argument('--data_path', type=str, default='../datasets/bpo/SCP/half/combined_BPO.json',
                       help='Path to the training data JSON file')
    parser.add_argument('--model_name', type=str, default='FacebookAI/roberta-large',
                       help='Pre-trained model name from HuggingFace (e.g., bert-base-uncased, FacebookAI/roberta-large, microsoft/deberta-v3-large)')
    parser.add_argument('--output_dir', type=str, default='logs',
                       help='Directory to save model outputs and logs')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size (actual batch size per GPU)')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Number of steps to accumulate gradients before updating (effective batch size = batch_size * accumulation_steps * num_gpus)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate for optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate for regularization')
    parser.add_argument('--gamma', type=float, default=2.0,
                       help='Gamma parameter for Focal Loss')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length for tokenization')
    args = parser.parse_args()

    set_seed(args.seed)
    
    # 모델명에서 폴더명 생성 (특수문자 제거)
    model_folder_name = args.model_name.replace('/', '_').replace('-', '_')
    param_str = (
        f"{model_folder_name}_"
        f"bs{args.batch_size}_acc{args.accumulation_steps}_lr{args.learning_rate}_drop{args.dropout_rate}_"
        f"ep{args.epochs}_seed{args.seed}_gm{args.gamma}"
    )
    output_folder = os.path.join(args.output_dir, param_str)
    os.makedirs(output_folder, exist_ok=True)
    
    effective_batch_size = args.batch_size * args.accumulation_steps
    if torch.cuda.device_count() > 1:
        effective_batch_size *= torch.cuda.device_count()
    
    print(f"Using model: {args.model_name}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Accumulation steps: {args.accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Output folder: {output_folder}")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # load and split data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Few shot Examples 필드 제외
    excluded_tasks = {'Few shot Examples'}
    
    class_counts = defaultdict(set)
    for item in data:
        for task, lbl in item['label'].items():
            if task not in excluded_tasks:  # None 값과 제외 태스크 제외
                class_counts[task].add(lbl)
    label2idx = {
        task: {lbl: idx for idx, lbl in enumerate(sorted(lbls))}
        for task, lbls in class_counts.items()}
    for item in data:
        item['label'] = {
            task: label2idx[task][lbl]
            for task, lbl in item['label'].items()
            if task not in excluded_tasks  # None 값과 제외 태스크 제외
        }

    train_data, temp = train_test_split(data, test_size=0.2, random_state=args.seed)
    val_data, test_data = train_test_split(temp, test_size=0.5, random_state=args.seed)


    num_classes = {task: len(lbl_map) for task, lbl_map in label2idx.items()}
    print(num_classes)
    
    # 명시적으로 cache_dir 지정하고 로컬 파일만 사용하도록 설정

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
    )

    
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(MultiTaskDataset(train_data, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader   = DataLoader(MultiTaskDataset(val_data, tokenizer, args.max_length),   batch_size=args.batch_size*2, collate_fn=collator)
    test_loader  = DataLoader(MultiTaskDataset(test_data, tokenizer, args.max_length),  batch_size=args.batch_size*2, collate_fn=collator)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskBERT(args.model_name, num_classes, args.dropout_rate, args.gamma).to(device)
    
    # GPU 병렬처리 설정
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    else:
        print(f"Using single GPU: {device}")

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    # 스케줄러 계산 시 accumulation steps 고려
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # training loop with batch accumulation
    best_f1, patience, no_improve = 0.0, 3, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        accumulated_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                **{k: batch[k] for k in batch if k.startswith('label_')}
            )
            
            loss = outputs['loss']
            if loss is not None:
                # accumulation steps로 나누어 평균 손실 계산
                loss = loss / args.accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                total_loss += loss.item()
                
                # accumulation steps만큼 쌓이면 업데이트
                if (step + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # 진행률 표시
                    progress_bar.set_postfix({
                        'loss': f'{accumulated_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
                    accumulated_loss = 0.0
        
        # 마지막 배치가 accumulation steps로 나누어떨어지지 않는 경우 처리
        if len(train_loader) % args.accumulation_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_loss / len(train_loader) * args.accumulation_steps
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Avg F1: {val_metrics['avg_f1']:.4f}")
        
        if val_metrics['avg_f1'] > best_f1:
            best_f1 = val_metrics['avg_f1']
            # DataParallel 사용 시 module을 통해 원본 모델에 접근
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(output_folder, 'best_model.pt'))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print('Early stopping.')
                break

    # test
    # DataParallel 사용 시 module을 통해 원본 모델에 로드
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(torch.load(os.path.join(output_folder, 'best_model.pt')))
    test_metrics = evaluate(model, test_loader, device)
    metrics_file = os.path.join(output_folder, 'test_metrics.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Batch size per GPU: {args.batch_size}\n")
        f.write(f"Accumulation steps: {args.accumulation_steps}\n")
        f.write(f"Effective batch size: {effective_batch_size}\n")
        f.write("="*50 + "\n")
        for task, m in test_metrics.items():
            if task in ['avg_f1','val_loss']: continue
            f.write(f"[{task}] Acc: {m['accuracy']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1']:.4f}\n")
        f.write(f"Overall Avg F1: {test_metrics['avg_f1']:.4f}\n")
    print(f"Metrics saved to {metrics_file}")

if __name__ == '__main__':
    main()

# 사용 예시:
# python train_with_accumulation.py --help  # 모든 옵션 확인
# python train_with_accumulation.py --model_name bert-base-uncased --epochs 5 --batch_size 8 --accumulation_steps 4
# python train_with_accumulation.py --model_name FacebookAI/roberta-large --batch_size 4 --accumulation_steps 8 --learning_rate 3e-5
# python train_with_accumulation.py --model_name microsoft/deberta-v3-large --batch_size 2 --accumulation_steps 16 --max_length 256
# python train_with_accumulation.py --model_name google/electra-large-discriminator --batch_size 8 --accumulation_steps 2 