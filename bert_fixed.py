"""
dnabert.py — DNABERT-2 fine-tuning v3 (overfitting fix)

ROOT CAUSE: Identical to XGBoost. repair.py duplicated real benign rows ~40x.
Random split placed near-identical copies in train/val/test.
DNABERT memorised sequences rather than splice biology → AUC 0.9961 is fake.
Train loss 0.007 at epoch 15 with val F1 0.9927 = classic memorisation.

FIXES:
  1. Deduplicate before splitting
  2. Chromosome-based split (same as train.py) — zero leakage
  3. Dropout increased: 0.1→0.3 in classifier, 0.1 in attention
  4. Weight decay 0.01→0.05
  5. Label smoothing 0.1 — prevents overconfident memorisation
  6. Early stopping on val AUC with patience=5 (was saving every AUC improvement)
  7. Max 10 epochs (was 15) — loss was already 0.007 meaning clear overfit
  8. Gradient accumulation kept at 4 for stability
  9. Learning rate halved: 2e-5→1e-5
  10. Explicit train-val AUC gap check per epoch
"""

import os, json, math, sys, re
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def patch_cache():
    base = Path.home() / ".cache" / "huggingface"
    NEW_REBUILD = '''\
    def rebuild_alibi_tensor(self, size, device=None):
        # WIN_ALIBI_REWRITE
        import math, torch
        n = self.num_attention_heads
        def get_slopes(n):
            def _p2(n):
                s = 2**(-(2**(-(math.log2(n)-3))))
                return [s * s**i for i in range(n)]
            if math.log2(n).is_integer(): return _p2(n)
            p = 2**math.floor(math.log2(n))
            return _p2(p) + get_slopes(2*p)[::2][:n-p]
        sl  = torch.tensor(get_slopes(n), dtype=torch.float32)
        pos = torch.arange(size, dtype=torch.float32)
        rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().unsqueeze(0)
        alibi = sl.view(-1,1,1) * -rel
        self.register_buffer("alibi", alibi.unsqueeze(1), persistent=False)
'''
    for p in base.rglob("bert_layers.py"):
        text = p.read_text(encoding="utf-8")
        if "WIN_ALIBI_REWRITE" not in text:
            m = re.search(
                r'    def rebuild_alibi_tensor\(self.*?(?=\n    def |\nclass |\Z)',
                text, flags=re.DOTALL)
            if m:
                text = text[:m.start()] + NEW_REBUILD + text[m.end():]
        lines = []
        for line in text.splitlines():
            s = line.strip()
            if any(s.startswith(x) for x in
                   ["import triton","from triton","import flash_attn","from flash_attn"]
                   ) and not s.startswith("#"):
                lines.append("# WIN_REMOVED: " + line)
            else:
                lines.append(line)
        p.write_text("\n".join(lines), encoding="utf-8")
    for p in base.rglob("flash_attn_triton.py"):
        p.write_text("# WIN_STUB\n", encoding="utf-8")
    for p in base.rglob("bert_padding.py"):
        text = p.read_text(encoding="utf-8")
        if "triton" in text and "WIN_REMOVED" not in text:
            lines = ["# WIN_REMOVED: " + l
                     if "triton" in l and not l.strip().startswith("#") else l
                     for l in text.splitlines()]
            p.write_text("\n".join(lines), encoding="utf-8")

patch_cache()

import importlib.util as _ilu
_dmu_spec = _ilu.find_spec("transformers.dynamic_module_utils")
_dmu_mod  = _ilu.module_from_spec(_dmu_spec)
_dmu_spec.loader.exec_module(_dmu_mod)
_orig_check = _dmu_mod.check_imports
def _patched_check(filename):
    try: return _orig_check(filename)
    except ImportError as e:
        if "triton" in str(e) or "flash_attn" in str(e): return []
        raise
_dmu_mod.check_imports = _patched_check
sys.modules["transformers.dynamic_module_utils"] = _dmu_mod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, average_precision_score, matthews_corrcoef
)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────
PARQUET_PATH  = "splice_windows_2.parquet"
MODEL_NAME    = "zhihan1996/DNABERT-2-117M"
WINDOW        = 200
MAX_LEN       = 128
BATCH_SIZE    = 32
ACCUM_STEPS   = 4          # effective batch = 128
EPOCHS        = 10         # was 15 — model was memorising by ep15
LR            = 1e-5       # was 2e-5 — slower learning = less memorisation
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.05       # was 0.01 — stronger L2
GRAD_CLIP     = 1.0
DROPOUT_CLS   = 0.3        # was 0.1 — more regularisation in classifier head
LABEL_SMOOTH  = 0.1        # was 0 — prevents confident memorisation of duplicates
PATIENCE      = 5          # early stopping: stop if val AUC doesn't improve for 5 epochs
SAVE_PATH     = "dnabert2_splice.pt"
RESULTS_PATH  = "dnabert2_results.json"
THRESH_PATH   = "dnabert2_threshold.json"
SEED          = 42

# Chromosome-based split (must match train.py)
TEST_CHROMS   = {"8", "21"}
VAL_CHROMS    = {"7", "X"}

torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
print(f"Device: {DEVICE} | AMP: {USE_AMP}")


def _safe_alibi(self, size, device=None):
    n = self.num_attention_heads
    def gs(n):
        def p2(n):
            s = 2**(-(2**(-(math.log2(n)-3))))
            return [s*s**i for i in range(n)]
        if math.log2(n).is_integer(): return p2(n)
        p = 2**math.floor(math.log2(n))
        return p2(p) + gs(2*p)[::2][:n-p]
    sl  = torch.tensor(gs(n), dtype=torch.float32)
    pos = torch.arange(size, dtype=torch.float32)
    rel = (pos.unsqueeze(0)-pos.unsqueeze(1)).abs().unsqueeze(0)
    alibi = sl.view(-1,1,1)*-rel
    self.register_buffer("alibi", alibi.unsqueeze(1), persistent=False)

def _patch_all():
    from transformers.models.bert.configuration_bert import BertConfig as StdCfg
    cache = Path.home()/".cache"/"huggingface"/"modules"
    for cfg_file in cache.rglob("configuration_bert.py"):
        mname = ".".join(cfg_file.with_suffix("").relative_to(cache).parts)
        if mname not in sys.modules:
            sp = _ilu.spec_from_file_location(mname, cfg_file)
            m  = _ilu.module_from_spec(sp); sp.loader.exec_module(m)
            m.BertConfig = StdCfg; sys.modules[mname] = m
        else:
            sys.modules[mname].BertConfig = StdCfg
    for mn, mod in list(sys.modules.items()):
        if "transformers_modules" in mn and hasattr(mod, "BertConfig"):
            mod.BertConfig = StdCfg
        if "bert_layers" in mn and hasattr(mod, "BertEncoder"):
            mod.BertEncoder.rebuild_alibi_tensor = _safe_alibi


# ── Label-smoothed cross entropy ──────────────────────────────────────
class SmoothCE(nn.Module):
    """
    Cross-entropy with label smoothing.
    Prevents the model from becoming overconfident on duplicate/near-duplicate
    rows, which was causing the train loss to collapse to 0.007.
    smoothing=0.1 means target 0→0.05, target 1→0.95 instead of hard 0/1.
    """
    def __init__(self, smoothing=LABEL_SMOOTH):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_cls   = logits.size(-1)
        log_p   = F.log_softmax(logits, dim=-1)
        smooth  = self.smoothing / (n_cls - 1)
        one_hot = torch.full_like(log_p, smooth)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(one_hot * log_p).sum(dim=-1).mean()


# ── Dataset ───────────────────────────────────────────────────────────
class SpliceDataset(Dataset):
    def __init__(self, df, tokenizer, window, max_len):
        self.labels = df["label"].values.astype(np.int64)
        rc, ac      = f"ref_seq_{window}", f"alt_seq_{window}"
        seqs = []
        for _, row in df.iterrows():
            ref = str(row[rc]).upper() if pd.notna(row.get(rc)) else "N"*window
            alt = str(row[ac]).upper() if pd.notna(row.get(ac)) else ref
            seqs.append(ref + " [SEP] " + alt)
        print(f"  Tokenising {len(seqs):,}...")
        self.enc = tokenizer(seqs, max_length=max_len, padding="max_length",
                             truncation=True, return_tensors="pt")

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {
            "input_ids":      self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels":         torch.tensor(self.labels[i], dtype=torch.long),
        }


# ── Model (more dropout) ──────────────────────────────────────────────
def load_model(device):
    _patch_all()
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _patch_all()
    base = None
    for cand in ["bert_layers.BertModel","modeling_bert.BertModel"]:
        try:
            cls  = get_class_from_dynamic_module(cand, MODEL_NAME)
            from transformers.models.bert.configuration_bert import BertConfig as StdCfg
            cls.config_class = StdCfg
            base = cls.from_pretrained(MODEL_NAME, config=config,
                                       trust_remote_code=True, low_cpu_mem_usage=False)
            print(f"Loaded base via {cand}")
            break
        except Exception as e:
            print(f"  {cand} failed: {e}")
    if base is None:
        raise RuntimeError("Cannot load BertModel")
    _patch_all()

    class DNABERTClassifier(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.base       = b
            self.drop       = nn.Dropout(DROPOUT_CLS)   # 0.3 — was 0.1
            self.classifier = nn.Linear(b.config.hidden_size, 2)

        def forward(self, input_ids, attention_mask):
            out    = self.base(input_ids=input_ids, attention_mask=attention_mask)
            pooled = (out[1] if (len(out)>=2 and out[1] is not None and out[1].dim()==2)
                      else out[0][:,0])
            return self.classifier(self.drop(pooled))

    return DNABERTClassifier(base).float().to(device)


# ── Chromosome split (mirrors train.py) ──────────────────────────────
def make_chrom_split(df):
    chrom   = df["chrom"].astype(str)
    te_mask = chrom.isin(TEST_CHROMS)
    va_mask = chrom.isin(VAL_CHROMS)
    tr_mask = ~te_mask & ~va_mask

    tr = df[tr_mask].reset_index(drop=True)
    va = df[va_mask].reset_index(drop=True)
    te = df[te_mask].reset_index(drop=True)

    print("\nChromosome split:")
    for name, sub in [("Train",tr),("Val",va),("Test",te)]:
        n0 = (sub["label"]==0).sum()
        n1 = (sub["label"]==1).sum()
        print(f"  {name:5} {len(sub):>6,}  benign={n0:,}  path={n1:,}")

    if len(va) < 100:
        print("  ⚠ Val too small — random 15% of train")
        idx   = np.random.RandomState(42).permutation(len(tr))
        n_val = max(int(len(tr)*0.15), 500)
        va    = df.iloc[tr_mask.values][idx[:n_val]].reset_index(drop=True)
        tr    = df.iloc[tr_mask.values][idx[n_val:]].reset_index(drop=True)
    if len(te) < 100:
        print("  ⚠ Test too small — random 15% of train")
        idx    = np.random.RandomState(42).permutation(len(tr))
        n_test = max(int(len(tr)*0.15), 500)
        te     = tr.iloc[idx[:n_test]].reset_index(drop=True)
        tr     = tr.iloc[idx[n_test:]].reset_index(drop=True)

    return tr, va, te


# ── MCC threshold ─────────────────────────────────────────────────────
def tune_threshold_mcc(probs, labels):
    best_t, best_mcc = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        mcc   = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t   = float(t)
    return best_t, best_mcc


# ── Train one epoch ───────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler_amp, criterion,
                accum=ACCUM_STEPS):
    model.train()
    total = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl  = batch["labels"].to(DEVICE)
        if USE_AMP:
            with torch.amp.autocast(device_type="cuda"):
                loss = criterion(model(ids, mask), lbl) / accum
            scaler_amp.scale(loss).backward()
        else:
            loss = criterion(model(ids, mask), lbl) / accum
            loss.backward()
        total += loss.item() * accum
        if (step+1) % accum == 0:
            if USE_AMP:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler_amp.step(optimizer); scaler_amp.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_p, all_l = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        if USE_AMP:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(ids, mask)
        else:
            logits = model(ids, mask)
        all_p.extend(torch.softmax(logits,-1)[:,1].cpu().numpy())
        all_l.extend(batch["labels"].numpy())
    p, l = np.array(all_p), np.array(all_l)
    pr   = (p >= threshold).astype(int)
    return {
        "auc":    roc_auc_score(l, p),
        "pr_auc": average_precision_score(l, p),
        "f1_path":f1_score(l, pr, pos_label=1, zero_division=0),
        "f1_ben": f1_score(l, pr, pos_label=0, zero_division=0),
        "mcc":    matthews_corrcoef(l, pr),
        "probs":  p, "labels": l,
    }


# ── Main ──────────────────────────────────────────────────────────────
def run():
    # Load and deduplicate
    df = pd.read_parquet(PARQUET_PATH)
    before = len(df)
    df     = df.drop_duplicates(subset=["chrom","position","ref","alt"])
    after  = len(df)
    if before > after:
        print(f"⚠ Removed {before-after:,} duplicate rows — {after:,} remain")

    window = WINDOW
    if f"ref_seq_{window}" not in df.columns:
        window = int([c for c in df.columns if c.startswith("ref_seq_")][0].split("_")[-1])

    train_df, val_df, test_df = make_chrom_split(df)

    n_ben = (train_df["label"]==0).sum()
    n_pat = (train_df["label"]==1).sum()
    ratio = n_pat / max(n_ben, 1)
    print(f"\nTrain: benign={n_ben:,}  pathogenic={n_pat:,}  ratio={ratio:.1f}x")

    # No class weights — data is balanced; label smoothing handles the rest
    criterion = SmoothCE(smoothing=LABEL_SMOOTH)
    print(f"Loss: label-smoothed CE  smoothing={LABEL_SMOOTH}  dropout={DROPOUT_CLS}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _patch_all()

    print("Loading model...")
    model = load_model(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nBuilding datasets...")
    train_ds = SpliceDataset(train_df, tokenizer, window, MAX_LEN)
    val_ds   = SpliceDataset(val_df,   tokenizer, window, MAX_LEN)
    test_ds  = SpliceDataset(test_df,  tokenizer, window, MAX_LEN)

    kw = dict(num_workers=0, pin_memory=USE_AMP)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False, **kw)

    no_decay  = ["bias","LayerNorm.weight"]
    optimizer = torch.optim.AdamW([
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params":[p for n,p in model.named_parameters() if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ], lr=LR)

    total_steps = EPOCHS * (len(train_loader) // ACCUM_STEPS)
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = int(WARMUP_RATIO*total_steps),
        num_training_steps = total_steps,
    )
    scaler_amp = torch.amp.GradScaler(device="cuda") if USE_AMP else None

    print(f"\nTraining {EPOCHS} epochs  lr={LR}  wd={WEIGHT_DECAY}  "
          f"label_smooth={LABEL_SMOOTH}  patience={PATIENCE}\n")

    best_auc    = 0.0
    best_epoch  = 0
    patience_ct = 0
    history     = []
    losses      = []

    for epoch in range(1, EPOCHS+1):
        tl  = train_epoch(model, train_loader, optimizer, scheduler, scaler_amp, criterion)
        vm  = evaluate(model, val_loader)

        # Also evaluate train set to measure gap
        tr_m = evaluate(model, DataLoader(train_ds, batch_size=BATCH_SIZE*2, shuffle=False, **kw))

        gap = tr_m["auc"] - vm["auc"]
        losses.append(tl)
        history.append(vm)

        flag = ""
        if vm["auc"] > best_auc:
            best_auc   = vm["auc"]
            best_epoch = epoch
            patience_ct = 0
            torch.save(model.state_dict(), SAVE_PATH)
            flag = "✓ saved"
        else:
            patience_ct += 1
            flag = f"patience {patience_ct}/{PATIENCE}"

        overfit_warn = " ⚠OVERFIT" if gap > 0.05 else (" ⚡mild" if gap > 0.02 else "")

        print(f"Ep {epoch:02d}/{EPOCHS}  loss={tl:.4f}  "
              f"train_auc={tr_m['auc']:.4f}  val_auc={vm['auc']:.4f}  "
              f"gap={gap:+.4f}{overfit_warn}  "
              f"f1_ben={vm['f1_ben']:.4f}  mcc={vm['mcc']:.4f}  {flag}")

        if patience_ct >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
            break

    # Load best
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False))
    print(f"\nBest epoch: {best_epoch}  val AUC: {best_auc:.4f}")

    # Threshold on val
    vm_best    = evaluate(model, val_loader)
    best_t, best_mcc = tune_threshold_mcc(vm_best["probs"], vm_best["labels"])
    macro_f1   = (
        f1_score(vm_best["labels"],(vm_best["probs"]>=best_t).astype(int),pos_label=0,zero_division=0) +
        f1_score(vm_best["labels"],(vm_best["probs"]>=best_t).astype(int),pos_label=1,zero_division=0)
    ) / 2
    print(f"Threshold: {best_t:.2f}  MCC={best_mcc:.4f}  macro-F1={macro_f1:.4f}")
    json.dump({"threshold":best_t,"val_mcc":best_mcc,"val_macro_f1":macro_f1},
              open(THRESH_PATH,"w"), indent=2)

    # Test evaluation
    tm = evaluate(model, test_loader, threshold=best_t)
    print(f"\n── Test ──")
    print(f"AUC={tm['auc']:.4f}  PR-AUC={tm['pr_auc']:.4f}  MCC={tm['mcc']:.4f}")
    print(classification_report(
        tm["labels"],(tm["probs"]>=best_t).astype(int),
        target_names=["Benign","Pathogenic"]
    ))

    json.dump({
        "test_auc":       tm["auc"],
        "test_pr_auc":    tm["pr_auc"],
        "test_mcc":       tm["mcc"],
        "test_f1_benign": tm["f1_ben"],
        "test_f1_path":   tm["f1_path"],
        "best_val_auc":   best_auc,
        "best_epoch":     best_epoch,
        "threshold":      best_t,
        "model":          MODEL_NAME,
        "window":         window,
        "epochs_trained": epoch,
        "label_smoothing":LABEL_SMOOTH,
        "dropout":        DROPOUT_CLS,
        "weight_decay":   WEIGHT_DECAY,
        "split":          "chromosome-based (test=chr8,21  val=chr7,X)",
    }, open(RESULTS_PATH,"w"), indent=2)

    # Curves
    ep  = range(1, len(losses)+1)
    fig, axes = plt.subplots(1,4,figsize=(20,4))
    for ax,vals,title in zip(axes, [
        losses,
        [m["auc"]    for m in history],
        [m["pr_auc"] for m in history],
        [m["mcc"]    for m in history],
    ], ["Train Loss","Val AUC","Val PR-AUC","Val MCC"]):
        ax.plot(ep, vals, marker="o", markersize=3)
        ax.axvline(best_epoch, color="red", ls="--", alpha=0.5, label="best")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("dnabert2_curves.png", dpi=150); plt.close()
    print("\nDone.")


if __name__ == "__main__":
    run()