"""
api.py — Splice Variant Classifier v5
"""
import os, time, sys, json, math
from itertools import product
from typing import Optional, List
from contextlib import asynccontextmanager
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]  = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

XGB_MODEL_PATH  = "xgb_model.json"
XGB_THRESH_PATH = "xgb_threshold.json"
DNA_THRESH_PATH = "dnabert2_threshold.json"
DNABERT_PATH    = "dnabert2_splice.pt"
DNABERT_NAME    = "zhihan1996/DNABERT-2-117M"
WINDOW          = 200
K               = 6
VOCAB_SIZE      = 4 ** K
ENSEMBLE_XGB_W  = 0.4
ENSEMBLE_DNA_W  = 0.6
DEFAULT_THRESH  = 0.5

BASES       = ["A","T","G","C"]
VOCAB       = ["".join(p) for p in product(BASES, repeat=K)]
VOCAB_INDEX = {kmer: i for i, kmer in enumerate(VOCAB)}
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models: dict = {}

# ── PWM matrices ──────────────────────────────────────────────────────
_DONOR_PWM = {
    0: {"A": 0.36, "C": 0.36, "G": 0.47, "T": -0.98},
    1: {"A": 0.05, "C":-0.44, "G": 0.30, "T":  0.10},
    2: {"A": 0.52, "C":-0.54, "G": 0.18, "T": -0.06},
    3: {"A":-2.00, "C":-2.00, "G": 1.50, "T": -2.00},
    4: {"A":-2.00, "C":-2.00, "G":-2.00, "T":  1.50},
    5: {"A": 0.70, "C":-1.80, "G": 0.14, "T": -0.70},
    6: {"A": 0.29, "C":-0.69, "G": 0.32, "T": -0.12},
    7: {"A": 0.42, "C":-0.97, "G": 0.30, "T": -0.16},
    8: {"A": 0.35, "C":-0.43, "G": 0.39, "T": -0.29},
}
_ACCEPTOR_PWM = {i: {"A":-0.6,"C":0.5,"G":-0.6,"T":0.5} for i in range(12)}
_ACCEPTOR_PWM[12] = {"A": 1.50,"C":-2.0,"G":-2.0,"T":-2.00}
_ACCEPTOR_PWM[13] = {"A":-2.00,"C":-2.0,"G": 1.50,"T":-2.00}


def _pwm_score(seq: str, pwm: dict) -> float:
    n = len(pwm)
    if len(seq) < n: return -99.0
    return sum(pwm[i].get(seq[i], -2.0) for i in range(n))

def _pwm_to_prob(raw: float, pwm_len: int) -> float:
    return 1.0 / (1.0 + math.exp(-raw / (pwm_len * 0.25)))


def scan_sites(seq: str, site_type: str) -> list:
    """
    Scan for donor (GT/GC motif at positions 3-4 of 9-mer)
    or acceptor (AG/AC at positions 12-13 of 14-mer).
    Returns pos, prob, kmer, context (exact sequence with [kmer] brackets).
    """
    seq  = seq.upper()
    pwm  = _DONOR_PWM if site_type=="donor" else _ACCEPTOR_PWM
    wlen = 9           if site_type=="donor" else 14

    if site_type == "donor":
        motif_ok = lambda s: s[3:5] in ("GT","GC","AT")
    else:
        motif_ok = lambda s: s[12:14] in ("AG","AC")

    results = []
    for i in range(len(seq) - wlen + 1):
        kmer = seq[i:i+wlen]
        if not motif_ok(kmer): continue
        raw  = _pwm_score(kmer, pwm)
        prob = _pwm_to_prob(raw, wlen)
        if prob < 0.15: continue
        cs   = max(0, i-5)
        ce   = min(len(seq), i+wlen+5)
        ctx  = seq[cs:i] + "[" + kmer + "]" + seq[i+wlen:ce]
        results.append({"pos":i,"raw":raw,"prob":round(prob,4),"kmer":kmer,"context":ctx})
    return results


def _classify_mutation(ref: str, alt: str) -> dict:
    ref, alt = ref.upper(), alt.upper()
    r, a = len(ref), len(alt)
    if r==1 and a==1:
        mut_type = "SNV"
        if   ref=="G": mechanism=f"G\u2192{alt} at +1 position of GT donor dinucleotide. Invariant site (>99% conserved) \u2014 disruption is nearly always pathogenic."
        elif ref=="T": mechanism=f"T\u2192{alt} at +2 position of GT donor. Highly conserved; substitution typically abolishes donor recognition."
        elif ref=="A": mechanism=f"A\u2192{alt} at \u22122 position of AG acceptor. Conservation >95%; loss causes exon skipping or intron retention."
        else:          mechanism=f"{ref}\u2192{alt} SNV in splice region. Effect depends on local sequence context and distance from exon boundary."
        detail = f"SNV {ref}\u2192{alt}"
    elif r > a:
        mut_type = "Deletion"; n=r-a; fs="frameshift" if n%3!=0 else "in-frame"
        mechanism=f"{n}bp {fs} deletion \u2014 likely disrupts splice site geometry and reading frame."; detail=f"{n}bp deletion"
    elif r < a:
        mut_type = "Insertion"; n=a-r; fs="frameshift" if n%3!=0 else "in-frame"
        mechanism=f"{n}bp {fs} insertion \u2014 may create cryptic splice sites or destroy existing signals."; detail=f"{n}bp insertion"
    else:
        mut_type="MNV"; mechanism=f"{r}bp complex substitution \u2014 high risk of disrupting overlapping splice signals."; detail=f"{r}bp MNV"
    return {"mutation_type":mut_type,"detail":detail,"mechanism":mechanism}


def analyse_splice_sites(ref_seq, alt_seq, ref, alt):
    ref_seq = ref_seq.upper(); alt_seq = alt_seq.upper()
    center  = len(ref_seq)//2
    mut     = _classify_mutation(ref, alt)
    output  = []

    for site_type in ("donor","acceptor"):
        ref_map = {s["pos"]:s for s in scan_sites(ref_seq, site_type)}
        alt_map = {s["pos"]:s for s in scan_sites(alt_seq, site_type)}
        for pos in sorted(set(ref_map)|set(alt_map)):
            rs  = ref_map.get(pos); as_ = alt_map.get(pos)
            rp  = rs["prob"]  if rs  else 0.0
            ap  = as_["prob"] if as_ else 0.0
            delta = round(ap-rp, 4)
            rel   = pos - center

            disrupted = rp>=0.45 and ap<0.45
            created   = rp<0.30  and ap>=0.45
            cryptic   = (not disrupted) and (not created) and delta>0.08 and ap>=0.30

            if abs(delta)<0.04 and not disrupted and not created and not cryptic:
                continue

            pos_str = f"+{rel}" if rel>=0 else str(rel)
            if disrupted:
                reasoning=(f"Canonical {site_type} site at {pos_str} loses PWM score "
                           f"{rp:.3f}\u2192{ap:.3f} (\u0394{delta:+.3f}). {mut['mechanism']} "
                           f"Expected consequence: aberrant splicing \u2014 exon skipping or intron retention.")
            elif created:
                reasoning=(f"De novo {site_type} site at {pos_str}: score {rp:.3f}\u2192{ap:.3f} "
                           f"(\u0394{delta:+.3f}). Novel splice signal competes with canonical sites \u2014 "
                           f"alternative splicing or novel isoform expected.")
            elif cryptic:
                reasoning=(f"Latent {site_type} site at {pos_str} strengthened: "
                           f"{rp:.3f}\u2192{ap:.3f} (\u0394{delta:+.3f}). Cryptic activation \u2014 "
                           f"may cause aberrant splicing in a subset of transcripts.")
            else:
                d="weakened" if delta<0 else "strengthened"
                reasoning=(f"{site_type.capitalize()} site at {pos_str} {d}: "
                           f"{rp:.3f}\u2192{ap:.3f} (\u0394{delta:+.3f}). "
                           f"Below canonical threshold \u2014 monitor in combination with other variants.")

            output.append({
                "type":site_type,"position":rel,
                "ref_kmer":rs["kmer"]    if rs  else "",
                "alt_kmer":as_["kmer"]   if as_ else "",
                "ref_context":rs["context"] if rs  else "",
                "alt_context":as_["context"]if as_ else "",
                "ref_score":rp,"alt_score":ap,"delta":delta,
                "disrupted":disrupted,"created":created,"cryptic":cryptic,
                "reasoning":reasoning,
            })

    return sorted(output, key=lambda x:abs(x["delta"]), reverse=True)[:30], mut


# ── Feature helpers ───────────────────────────────────────────────────
def kmer_freq_vector(seq):
    seq = seq.upper(); vec=np.zeros(VOCAB_SIZE,dtype=np.float32); n=len(seq)-K+1
    if n<=0: return vec
    for i in range(n):
        km=seq[i:i+K]
        if km in VOCAB_INDEX: vec[VOCAB_INDEX[km]]+=1
    vec/=(n+1e-10); return vec

def variant_type_int(ref,alt):
    if len(ref)==1 and len(alt)==1: return 0
    if len(ref)<len(alt): return 1
    if len(ref)>len(alt): return 2
    return 3

def bio_features(ref,alt,ref_seq):
    r,a=ref.upper(),alt.upper(); seq=ref_seq.upper() if ref_seq else ""; f=np.zeros(9,dtype=np.float32)
    if len(r)==1 and len(a)==1:
        f[0]=float(r=="G"and a!="G"); f[1]=float(r=="T"and a!="T")
        f[2]=float(r=="A"and a!="A"); f[3]=float(r=="G"and a!="G")
    f[4]=float("GT"in r); f[5]=float("AG"in r)
    f[6]=float("GT"in r and "GT"not in a); f[7]=float("AG"in r and "AG"not in a)
    if seq: f[8]=(seq.count("G")+seq.count("C"))/max(len(seq),1)
    return f

def apply_mutation(win,center,ref,alt): return win[:center]+alt+win[center+len(ref):]

def build_feature_vector(ref,alt,ref_seq,alt_seq):
    rv=kmer_freq_vector(ref_seq) if ref_seq else np.zeros(VOCAB_SIZE,np.float32)
    av=kmer_freq_vector(alt_seq) if alt_seq else np.zeros(VOCAB_SIZE,np.float32)
    d=av-rv; mag=np.linalg.norm(d); direction=d/(mag+1e-10)
    return np.concatenate([rv,av,direction,np.array([mag],np.float32),
                           np.array([variant_type_int(ref,alt),len(ref),len(alt)],np.float32),
                           bio_features(ref,alt,ref_seq or "")])

def named_dmatrix(feat):
    cols=[f"f_{i}" for i in range(len(feat))]
    return xgb.DMatrix(pd.DataFrame(feat.reshape(1,-1),columns=cols))


# ── Schemas ───────────────────────────────────────────────────────────
class VariantRequest(BaseModel):
    chrom:str; position:int; ref:str; alt:str
    ref_seq:Optional[str]=None; alt_seq:Optional[str]=None
    def get_alt_seq(self):
        if self.alt_seq: return self.alt_seq
        if self.ref_seq:
            c=len(self.ref_seq)//2; return apply_mutation(self.ref_seq,c,self.ref,self.alt)
        return None

class PredictionResponse(BaseModel):
    chrom:str; position:int; ref:str; alt:str
    model:str; probability:float; prediction:str
    confidence:str; latency_ms:float; threshold_used:float
    mutation_type:str; mechanism:str

class SiteDetail(BaseModel):
    type:str; position:int
    ref_kmer:str; alt_kmer:str
    ref_context:str; alt_context:str   # exact sequence with [kmer] brackets
    ref_score:float; alt_score:float; delta:float
    disrupted:bool; created:bool; cryptic:bool; reasoning:str

class SpliceAnalysisResponse(BaseModel):
    chrom:str; position:int; ref:str; alt:str
    mutation_type:str; mutation_detail:str; mutation_mechanism:str
    sites_found:int
    disrupted_donors:int; disrupted_acceptors:int
    created_donors:int;   created_acceptors:int
    cryptic_donors:int;   cryptic_acceptors:int
    max_disruption:float; pathogenicity_signal:str; summary:str
    sites:List[SiteDetail]; latency_ms:float


# ── DNABERT loader ────────────────────────────────────────────────────
def _safe_alibi(self,size,device=None):
    n=self.num_attention_heads
    def gs(n):
        def p2(n):
            s=2**(-(2**(-(math.log2(n)-3)))); return [s*s**i for i in range(n)]
        if math.log2(n).is_integer(): return p2(n)
        p=2**math.floor(math.log2(n)); return p2(p)+gs(2*p)[::2][:n-p]
    sl=torch.tensor(gs(n),dtype=torch.float32)
    pos=torch.arange(size,dtype=torch.float32)
    rel=(pos.unsqueeze(0)-pos.unsqueeze(1)).abs().unsqueeze(0)
    self.register_buffer("alibi",(sl.view(-1,1,1)*-rel).unsqueeze(1),persistent=False)

def _patch_all():
    import importlib.util as ilu
    from transformers.models.bert.configuration_bert import BertConfig as StdCfg
    cache=Path.home()/".cache"/"huggingface"/"modules"
    for cf in cache.rglob("configuration_bert.py"):
        mn=".".join(cf.with_suffix("").relative_to(cache).parts)
        if mn not in sys.modules:
            sp=ilu.spec_from_file_location(mn,cf); m=ilu.module_from_spec(sp); sp.loader.exec_module(m)
            m.BertConfig=StdCfg; sys.modules[mn]=m
        else: sys.modules[mn].BertConfig=StdCfg
    for mn,mod in list(sys.modules.items()):
        if "transformers_modules" in mn and hasattr(mod,"BertConfig"): mod.BertConfig=StdCfg
        if "bert_layers" in mn and hasattr(mod,"BertEncoder"): mod.BertEncoder.rebuild_alibi_tensor=_safe_alibi

def _load_dnabert():
    from transformers import AutoTokenizer,AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    _patch_all()
    tok=AutoTokenizer.from_pretrained(DNABERT_NAME,trust_remote_code=True); _patch_all()
    cfg=AutoConfig.from_pretrained(DNABERT_NAME,trust_remote_code=True); _patch_all()
    Cls=get_class_from_dynamic_module("bert_layers.BertModel",DNABERT_NAME)
    base=Cls.from_pretrained(DNABERT_NAME,config=cfg,trust_remote_code=True,low_cpu_mem_usage=False); _patch_all()
    class DNABERTClassifier(nn.Module):
        def __init__(self,b):
            super().__init__(); self.base=b; self.drop=nn.Dropout(0.1)
            self.classifier=nn.Linear(b.config.hidden_size,2)
        def forward(self,input_ids,attention_mask):
            out=self.base(input_ids=input_ids,attention_mask=attention_mask)
            pooled=out[1] if(len(out)>=2 and out[1] is not None and out[1].dim()==2)else out[0][:,0]
            return self.classifier(self.drop(pooled))
    clf=DNABERTClassifier(base).float().to(DEVICE)
    st=torch.load(DNABERT_PATH,map_location=DEVICE,weights_only=False)
    mis,unx=clf.load_state_dict(st,strict=False)
    print(f"  DNABERT: missing={len(mis)} unexpected={len(unx)}"); clf.eval(); return clf,tok


# ── Helpers ───────────────────────────────────────────────────────────
def _confidence(prob,thresh):
    d=abs(prob-thresh); return "High" if d>=0.35 else "Medium" if d>=0.15 else "Low"

def _load_thresh(path,label):
    if os.path.exists(path):
        t=json.load(open(path))["threshold"]; print(f"  {label} threshold: {t:.4f}"); return t
    print(f"  WARNING: {path} not found, using {DEFAULT_THRESH}"); return DEFAULT_THRESH

def _dna_prob(ref_seq,alt_seq):
    enc=models["tokenizer"](ref_seq+" [SEP] "+alt_seq,max_length=128,
                            padding="max_length",truncation=True,return_tensors="pt")
    with torch.no_grad():
        logits=models["dnabert"](enc["input_ids"].to(DEVICE),enc["attention_mask"].to(DEVICE))
    return float(torch.softmax(logits,-1)[0,1].cpu())


# ── Lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app:FastAPI):
    if os.path.exists(XGB_MODEL_PATH):
        m=xgb.Booster(); m.load_model(XGB_MODEL_PATH)
        models["xgb"]=m; models["xgb_thresh"]=_load_thresh(XGB_THRESH_PATH,"XGB")
        print("  XGBoost ready")
    else: print(f"WARNING: {XGB_MODEL_PATH} not found")
    if os.path.exists(DNABERT_PATH):
        try:
            clf,tok=_load_dnabert()
            models["dnabert"]=clf; models["tokenizer"]=tok
            models["dna_thresh"]=_load_thresh(DNA_THRESH_PATH,"DNABERT")
            print(f"  DNABERT-2 ready on {DEVICE}")
        except Exception as e:
            import traceback; print(f"  DNABERT failed: {e}"); traceback.print_exc()
    else: print(f"WARNING: {DNABERT_PATH} not found")
    yield; models.clear()


# ── App ───────────────────────────────────────────────────────────────
app=FastAPI(title="Splice Variant Classifier",version="5.0.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,
                   allow_methods=["*"],allow_headers=["*"])

def _pred_response(req,model_name,prob,thresh):
    mut=_classify_mutation(req.ref,req.alt)
    return PredictionResponse(
        chrom=req.chrom,position=req.position,ref=req.ref,alt=req.alt,
        model=model_name,probability=round(prob,6),
        prediction="Pathogenic" if prob>=thresh else "Benign",
        confidence=_confidence(prob,thresh),threshold_used=thresh,
        mutation_type=mut["mutation_type"],mechanism=mut["mechanism"],latency_ms=0.0)

@app.post("/predict/xgb",response_model=PredictionResponse)
async def predict_xgb(req:VariantRequest):
    if "xgb" not in models: raise HTTPException(503,"XGBoost not loaded")
    t0=time.perf_counter()
    feat=build_feature_vector(req.ref,req.alt,req.ref_seq,req.get_alt_seq())
    prob=float(models["xgb"].predict(named_dmatrix(feat))[0])
    r=_pred_response(req,"XGBoost",prob,models.get("xgb_thresh",DEFAULT_THRESH))
    r.latency_ms=round((time.perf_counter()-t0)*1000,2); return r

@app.post("/predict/dnabert",response_model=PredictionResponse)
async def predict_dnabert(req:VariantRequest):
    if "dnabert" not in models: raise HTTPException(503,"DNABERT-2 not loaded")
    t0=time.perf_counter()
    rs=req.ref_seq or("N"*WINDOW); as_=req.get_alt_seq() or rs
    prob=_dna_prob(rs,as_)
    r=_pred_response(req,"DNABERT-2",prob,models.get("dna_thresh",DEFAULT_THRESH))
    r.latency_ms=round((time.perf_counter()-t0)*1000,2); return r

@app.post("/predict/ensemble",response_model=PredictionResponse)
async def predict_ensemble(req:VariantRequest):
    if "xgb" not in models and "dnabert" not in models: raise HTTPException(503,"No models loaded")
    t0=time.perf_counter(); probs=[]; weights=[]
    xt=models.get("xgb_thresh",DEFAULT_THRESH); dt=models.get("dna_thresh",DEFAULT_THRESH)
    if "xgb" in models:
        feat=build_feature_vector(req.ref,req.alt,req.ref_seq,req.get_alt_seq())
        probs.append(float(models["xgb"].predict(named_dmatrix(feat))[0])); weights.append(ENSEMBLE_XGB_W)
    if "dnabert" in models:
        rs=req.ref_seq or("N"*WINDOW); as_=req.get_alt_seq() or rs
        probs.append(_dna_prob(rs,as_)); weights.append(ENSEMBLE_DNA_W)
    w=np.array(weights)/sum(weights); prob=float(np.dot(probs,w))
    t_ens=(xt*(ENSEMBLE_XGB_W/sum(weights))+dt*(ENSEMBLE_DNA_W/sum(weights))) if len(probs)>1 else xt
    r=_pred_response(req,"Ensemble (XGB+DNABERT-2)",prob,round(t_ens,4))
    r.latency_ms=round((time.perf_counter()-t0)*1000,2); return r

@app.post("/predict/splice_sites",response_model=SpliceAnalysisResponse)
async def predict_splice_sites(req:VariantRequest):
    if not req.ref_seq: raise HTTPException(400,"ref_seq required")
    t0=time.perf_counter()
    alt_seq=req.get_alt_seq() or apply_mutation(req.ref_seq,len(req.ref_seq)//2,req.ref,req.alt)
    sites,mut=analyse_splice_sites(req.ref_seq,alt_seq,req.ref,req.alt)
    dd=sum(1 for s in sites if s["disrupted"] and s["type"]=="donor")
    da=sum(1 for s in sites if s["disrupted"] and s["type"]=="acceptor")
    cd=sum(1 for s in sites if s["created"]   and s["type"]=="donor")
    ca=sum(1 for s in sites if s["created"]   and s["type"]=="acceptor")
    kd=sum(1 for s in sites if s["cryptic"]   and s["type"]=="donor")
    ka=sum(1 for s in sites if s["cryptic"]   and s["type"]=="acceptor")
    mx=max((abs(s["delta"]) for s in sites),default=0.0)
    if   dd+da>=1 and cd+ca+kd+ka>=1: sig="Strong pathogenic — canonical site lost + cryptic/new site gained"
    elif dd+da>=2:                     sig="Strong pathogenic — multiple canonical sites disrupted"
    elif dd+da==1:                     sig="Moderate — canonical splice site disrupted"
    elif cd+ca>=1:                     sig="Moderate — new splice site created"
    elif kd+ka>=1:                     sig="Weak — cryptic splice site activated"
    elif mx>0.08:                      sig="Low — minor splicing score change"
    else:                              sig="Benign — no significant splicing impact"
    parts=[]
    if dd: parts.append(f"{dd} donor(s) disrupted")
    if da: parts.append(f"{da} acceptor(s) disrupted")
    if cd: parts.append(f"{cd} new donor(s)")
    if ca: parts.append(f"{ca} new acceptor(s)")
    if kd: parts.append(f"{kd} cryptic donor(s)")
    if ka: parts.append(f"{ka} cryptic acceptor(s)")
    return SpliceAnalysisResponse(
        chrom=req.chrom,position=req.position,ref=req.ref,alt=req.alt,
        mutation_type=mut["mutation_type"],mutation_detail=mut["detail"],
        mutation_mechanism=mut["mechanism"],sites_found=len(sites),
        disrupted_donors=dd,disrupted_acceptors=da,created_donors=cd,created_acceptors=ca,
        cryptic_donors=kd,cryptic_acceptors=ka,max_disruption=round(mx,4),
        pathogenicity_signal=sig,summary="; ".join(parts) if parts else "No splice site changes detected",
        sites=[SiteDetail(**s) for s in sites],
        latency_ms=round((time.perf_counter()-t0)*1000,2))

@app.get("/health")
async def health():
    return {"status":"ok","xgb":"xgb" in models,"dnabert":"dnabert" in models,
            "xgb_threshold":models.get("xgb_thresh",DEFAULT_THRESH),
            "dna_threshold":models.get("dna_thresh",DEFAULT_THRESH),"device":str(DEVICE)}

@app.get("/")
async def root():
    return {"endpoints":["/predict/xgb","/predict/dnabert","/predict/ensemble",
                          "/predict/splice_sites","/health","/docs"]}

if __name__=="__main__":
    import uvicorn; uvicorn.run("api:app",host="0.0.0.0",port=8000,reload=True)