#!/usr/bin/env python3
import os, sys, json, math, argparse, textwrap, random, zipfile, io
from pathlib import Path
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# -------------------------------
# Utility helpers
# -------------------------------

def to_dtype(s):
    s = s.lower()
    if s in ("fp16","float16","16"):
        return torch.float16
    if s in ("bf16","bfloat16"):
        return torch.bfloat16
    return torch.float32

@torch.no_grad()
def compute_attentions(model, tok, text, max_len=256, device="cuda"):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k:v.to(device) for k,v in enc.items()}
    out = model(**enc, output_attentions=True, return_dict=True)
    # out.attentions: tuple(L) of (B, H, S, S)
    return enc, out

def head_entropy(attn):  # attn: (B,H,S,S)
    # softmax already; compute entropy over keys for each query token, then mean
    # small epsilon to avoid log(0)
    eps = 1e-12
    p = attn.clamp_min(eps)
    H = -(p * p.log()).sum(dim=-1)  # (B,H,S)
    return H.mean(dim=(0,2))        # (H,)

def head_concentration(attn):
    # Return mean attention weight to first and last tokens (CLS/sink behavior)
    B,H,S,_ = attn.shape
    conc_first = attn[..., 0].mean(dim=(0,2))   # (H,)
    conc_last  = attn[..., -1].mean(dim=(0,2))  # (H,)
    return conc_first, conc_last

def last_layer_diversity(attn_tuple):
    # cosine-sim matrix over flattened maps, then |off-diag| mean
    last = attn_tuple[-1]          # (B,H,S,S)
    B,H,S,_ = last.shape
    flat = last.reshape(B,H,-1)
    flat = F.normalize(flat, p=2, dim=-1)
    sim = torch.einsum("bhi,bhj->bhj", flat, flat)  # (B,H,H)
    eye = torch.eye(H, device=sim.device).unsqueeze(0)
    off = (sim - eye).abs()
    return off.mean().item()

def cross_entropy_loss(logits, labels):
    # shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    return F.cross_entropy(shift_logits, shift_labels)

class OProjHeadMask:
    """Forward-input hook that zeroes a specific head slice at a given layer's o_proj."""
    def __init__(self, model, layer_idx, head_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.hook = None
        self.start = None
        self.end = None

    def _find(self):
        attn = self.model.model.layers[self.layer_idx].self_attn
        # head_dim: try to use attribute; otherwise derive
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            # derive via hidden_size // num_heads
            num_heads = attn.num_key_value_groups * (attn.config.num_attention_heads // attn.config.num_key_value_heads)
            head_dim = attn.o_proj.in_features // num_heads
        self.start = self.head_idx * head_dim
        self.end   = (self.head_idx + 1) * head_dim
        return attn.o_proj

    def __enter__(self):
        layer = self._find()
        def pre_hook(module, inputs):
            (x,) = inputs  # x: (..., H*D)
            if x is None:
                return inputs
            x = x.clone()
            x[..., self.start:self.end] = 0
            return (x,)
        self.hook = layer.register_forward_pre_hook(pre_hook, with_kwargs=False)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.hook is not None:
            self.hook.remove()
        return False

@torch.no_grad()
def abl_delta_loss(model, tok, text, layer_idx, head_idx, max_len=256, device="cuda"):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k:v.to(device) for k,v in enc.items()}
    base = model(**enc, output_attentions=False, return_dict=True)
    base_loss = cross_entropy_loss(base.logits, enc["input_ids"]).item()
    with OProjHeadMask(model, layer_idx, head_idx):
        pert = model(**enc, output_attentions=False, return_dict=True)
        pert_loss = cross_entropy_loss(pert.logits, enc["input_ids"]).item()
    return max(0.0, pert_loss - base_loss)

def save_heatmap(tensor_2d, path, title=""):
    plt.figure(figsize=(6,5))
    plt.imshow(tensor_2d, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# -------------------------------
# Main analyses
# -------------------------------

def run_quick(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_dtype(args.dtype)

    # Model
    qconf = None
    if args.bits4:
        qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=False,
                                   bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=(torch.bfloat16 if dtype==torch.bfloat16 else torch.float16))
    tok = AutoTokenizer.from_pretrained(args.base_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_id,
        torch_dtype=dtype if qconf is None else None,
        device_map="auto" if qconf else None,
        quantization_config=qconf
    )
    if args.adapter_dir and _HAS_PEFT:
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    model.eval()
    model.to(device)

    # Sample generation
    if args.prompt:
        enc = tok(args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model.generate(**enc, max_new_tokens=40)
        print("— Generation sample complete —")

    # Probe
    text = args.probe or "The product launch was delayed due to supply chain issues. Customers were notified via email and a new timeline was agreed."
    enc, out = compute_attentions(model, tok, text, max_len=args.max_len, device=device)
    L = len(out.attentions)
    B,H,S,_ = out.attentions[-1].shape
    print(f"[probe] layers={L}, heads={H}, tokens={S}")

    # Diversity
    div = last_layer_diversity(out.attentions)
    print(f"📊 Last-layer diversity (mean |off-diag|): {div:.4f}")

    # Per-head stats last layer
    last = out.attentions[-1]  # (B,H,S,S)
    ent = head_entropy(last).cpu().numpy()                 # (H,)
    c1, cL = head_concentration(last)
    c1 = c1.cpu().numpy(); cL = cL.cpu().numpy()

    # Ablations (last layer)
    layer_idx = L - 1
    deltas = []
    for h in range(H):
        d = abl_delta_loss(model, tok, text, layer_idx, h, max_len=args.max_len, device=device)
        deltas.append((h, d))
    deltas.sort(key=lambda x: x[1], reverse=True)

    # Save one heatmap for top head
    top_h = deltas[0][0]
    mean_map = last[:, top_h].mean(dim=0).cpu().numpy()  # (S,S)
    save_heatmap(mean_map, str(Path(args.out_dir)/"last_layer_head_top.png"),
                 f"Layer {layer_idx} Head {top_h} (mean over batch)")

    # Report
    rep = {
        "layers": L, "heads": H, "tokens": int(S),
        "diversity_abs_offdiag_mean": float(div),
        "top10_by_delta_loss": [{"head": h, "delta_loss": float(d)} for h,d in deltas[:10]],
        "entropy": [float(x) for x in ent.tolist()],
        "conc_first": [float(x) for x in c1.tolist()],
        "conc_last": [float(x) for x in cL.tolist()],
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir)/"quick_report.json","w") as f:
        json.dump(rep, f, indent=2)
    print("✅ quick mode done")
    print(f"  JSON: {Path(args.out_dir)/'quick_report.json'}")
    print(f"  Heatmap: {Path(args.out_dir)/'last_layer_head_top.png'}")

def iter_layers(model):
    return range(len(model.model.layers))

def run_deep(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_dtype(args.dtype)

    qconf = None
    if args.bits4:
        qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=False,
                                   bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=(torch.bfloat16 if dtype==torch.bfloat16 else torch.float16))
    tok = AutoTokenizer.from_pretrained(args.base_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_id,
        torch_dtype=dtype if qconf is None else None,
        device_map="auto" if qconf else None,
        quantization_config=qconf
    )
    if args.adapter_dir and _HAS_PEFT:
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    model.eval()
    model.to(device)

    text = args.probe or "We will consolidate vendors to reduce cost of goods, renegotiate SLAs, and improve gross margin next quarter."
    enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
    with torch.no_grad():
        out = model(**enc, output_attentions=True, return_dict=True)

    L = len(out.attentions)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    rows = ["layer,head,delta_loss,entropy,conc_first,conc_last"]
    summary = {}
    for layer_idx in reversed(list(range(L))):  # deepest first
        attn = out.attentions[layer_idx]  # (B,H,S,S)
        B,H,S,_ = attn.shape
        ent = head_entropy(attn).cpu().numpy()
        c1,cL = head_concentration(attn)
        c1 = c1.cpu().numpy(); cL = cL.cpu().numpy()

        deltas = []
        for h in range(H):
            d = abl_delta_loss(model, tok, text, layer_idx, h, max_len=args.max_len, device=device)
            deltas.append((h,d))
        deltas.sort(key=lambda x: x[1], reverse=True)

        # save heatmap for top head
        top_h = deltas[0][0]
        heat = attn[:, top_h].mean(dim=0).cpu().numpy()
        save_heatmap(heat, str(Path(args.out_dir)/f"layer_{layer_idx}_head_{top_h}.png"),
                     f"Layer {layer_idx} Head {top_h}")

        # rows
        for h,d in deltas:
            rows.append(f"{layer_idx},{h},{d:.6f},{float(ent[h]):.6f},{float(c1[h]):.6f},{float(cL[h]):.6f}")

        summary[layer_idx] = {
            "top": [{"head": int(h), "delta_loss": float(d)} for h,d in deltas[:5]]
        }
        print(f"  Layer {layer_idx}: saved heatmap for head {top_h}")

    with open(Path(args.out_dir)/"ablation_all_layers.csv","w") as f:
        f.write("\n".join(rows))
    with open(Path(args.out_dir)/"master_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    # zip artifacts
    zip_path = Path(args.out_dir).with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in Path(args.out_dir).glob("*.png"):
            zf.write(p, p.name)
        for p in [Path(args.out_dir)/"ablation_all_layers.csv", Path(args.out_dir)/"master_summary.json"]:
            zf.write(p, p.name)

    print("✅ deep mode done")
    print(f"  CSV : {Path(args.out_dir)/'ablation_all_layers.csv'}")
    print(f"  JSON: {Path(args.out_dir)/'master_summary.json'}")
    print(f"  ZIP : {zip_path}")

def enterprise_prompts():
    return [
        "Summarize the Q3 revenue growth drivers for a B2B SaaS with usage-based pricing.",
        "Draft a post-mortem for a failed cloud migration highlighting remediation steps.",
        "Explain a vendor consolidation plan to cut COGS by 5% over two quarters.",
        "Outline an incident response runbook for a P1 service outage.",
        "Design a KPI dashboard for enterprise customer health scoring.",
        "Write an email to renegotiate SLAs with a strategic supplier.",
        "Propose a cost optimization roadmap for a cloud-native data platform.",
        "Create a change management plan for a CRM system migration.",
        "Summarize a root-cause analysis of a billing system regression.",
        "Draft an executive update on gross margin improvement initiatives."
    ]

def run_enterprise(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_dtype(args.dtype)
    qconf = None
    if args.bits4:
        qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=False,
                                   bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=(torch.bfloat16 if dtype==torch.bfloat16 else torch.float16))
    tok = AutoTokenizer.from_pretrained(args.base_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_id,
        torch_dtype=dtype if qconf is None else None,
        device_map="auto" if qconf else None,
        quantization_config=qconf
    )
    if args.adapter_dir and _HAS_PEFT:
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    model.eval()
    model.to(device)

    # candidates: try to read from deep summary if provided
    candidates = []
    if args.summary_json and Path(args.summary_json).exists():
        data = json.loads(Path(args.summary_json).read_text())
        # grab a few with 'construct' likely true (top entries)
        for layer_str, info in data.items():
            layer = int(layer_str)
            for d in info.get("top", [])[:2]:
                candidates.append((layer, int(d["head"])))
    # fallback: use last layer heads 0..4
    if not candidates:
        with torch.no_grad():
            enc, out = compute_attentions(model, tok,
                "Enterprise customers escalated due to missed SLAs; propose mitigation and communication plan.",
                max_len=args.max_len, device=device)
        L = len(out.attentions); H = out.attentions[-1].shape[1]
        candidates = [(L-1, h) for h in range(min(5, H))]

    prompts = enterprise_prompts()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # stability + ablations
    records = []
    for (layer, head) in candidates:
        entropies = []
        conc1s, concLs = [], []
        for p in prompts:
            enc, out = compute_attentions(model, tok, p, max_len=args.max_len, device=device)
            attn = out.attentions[layer]  # (B,H,S,S)
            eh = head_entropy(attn)[head].item()
            c1, cL = head_concentration(attn)
            entropies.append(eh)
            conc1s.append(c1[head].item()); concLs.append(cL[head].item())

        # ablation on concatenated enterprise text
        big_text = " ".join(prompts)[: min(1024, args.max_len*4)]
        delta = abl_delta_loss(model, tok, big_text, layer, head, max_len=args.max_len, device=device)

        # heatmap for one prompt
        enc, out = compute_attentions(model, tok, prompts[0], max_len=args.max_len, device=device)
        hm = out.attentions[layer][:, head].mean(dim=0).cpu().numpy()
        save_heatmap(hm, str(Path(args.out_dir)/f"layer_{layer}_head_{head}.png"),
                     f"Enterprise drilldown L{layer}H{head}")

        rec = {
            "layer": layer, "head": head,
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std":  float(np.std(entropies)),
            "conc_first_mean": float(np.mean(conc1s)),
            "conc_last_mean":  float(np.mean(concLs)),
            "delta_loss": float(delta),
        }
        records.append(rec)
        print(f"  L{layer}H{head}: Δloss={rec['delta_loss']:.4f}, "
              f"entropy {rec['entropy_mean']:.3f}±{rec['entropy_std']:.3f}, "
              f"conc_first {rec['conc_first_mean']:.3f}, conc_last {rec['conc_last_mean']:.3f}")

    # write artifacts
    with open(Path(args.out_dir)/"candidate_evidence.json","w") as f:
        json.dump(records, f, indent=2)
    with open(Path(args.out_dir)/"candidate_evidence.csv","w") as f:
        f.write("layer,head,delta_loss,entropy_mean,entropy_std,conc_first_mean,conc_last_mean\n")
        for r in records:
            f.write(f"{r['layer']},{r['head']},{r['delta_loss']:.6f},"
                    f"{r['entropy_mean']:.6f},{r['entropy_std']:.6f},"
                    f"{r['conc_first_mean']:.6f},{r['conc_last_mean']:.6f}\n")

    # zip heatmaps
    zip_path = Path(args.out_dir)/"drilldown_heatmaps_enterprise.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in Path(args.out_dir).glob("*.png"):
            zf.write(p, p.name)

    print("✅ enterprise mode done")
    print(f"  Evidence CSV : {Path(args.out_dir)/'candidate_evidence.csv'}")
    print(f"  Evidence JSON: {Path(args.out_dir)/'candidate_evidence.json'}")
    print(f"  Heatmaps ZIP : {zip_path}")

# -------------------------------
# CLI
# -------------------------------

def main():
    p = argparse.ArgumentParser(description="CBT head analysis (TinyLlama + LoRA)")
    p.add_argument("--base_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter_dir", type=str, default="", help="LoRA adapter dir (optional)")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32","float16","bfloat16","fp16","bf16"])
    p.add_argument("--bits4", action="store_true", help="Use 4-bit quantization")
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--probe", type=str, default="")
    p.add_argument("--mode", type=str, default="quick", choices=["quick","deep","enterprise"])
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--summary_json", type=str, default="", help="deep summary JSON to seed enterprise candidates")
    args = p.parse_args()

    if args.mode == "quick":
        run_quick(args)
    elif args.mode == "deep":
        run_deep(args)
    else:
        run_enterprise(args)

if __name__ == "__main__":
    main()
