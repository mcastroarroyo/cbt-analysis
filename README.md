Here’s a tight recap of the **exact TinyLlama architecture changes** we made for CBT so the training script can “see” and penalize redundant heads:

# What we changed (and where) -> CBTtinyllamaMod

1. LlamaAttention.forward — **surface raw attention weights**

* Kept the math the same; added support to **return `attn_weights`** when `output_attentions=True`.
* No param/shape change; only exposes what was already computed.
* Return now: `(attn_output, attn_weights)` when requested, else `(attn_output, None)`.

2. LlamaDecoderLayer.forward — **thread the flag + bubble up weights**

* Accepts `output_attentions` and **passes it into `self_attn`**.
* **Appends the per-layer `self_attn_weights` to the layer outputs** when requested.
* Keeps cache behavior unchanged; tuples stay backward-compatible.

3. LlamaModel.forward — **collect per-layer attentions**

* Adds/threads the standard flags:

  * `output_attentions`, `output_hidden_states`, `use_cache`, `return_dict`.
* **Creates rotary `position_embeddings` once** and reuses in each layer (minor efficiency win).
* **Accumulates `all_self_attns`** (a tuple of `(num_heads, seq, seq)` per layer) when requested.
* Returns a **`BaseModelOutputWithPast`** that now includes `.attentions`.

4. LlamaForCausalLM.forward — **propagate flags to the backbone**

* Forwards the same flags into `self.model(...)`.
* Loss/logits logic unchanged.
* Returns a **`CausalLMOutputWithPast`** that forwards through `.attentions`.

5. Safety/robustness shims (to avoid the HF wrapper crash)

* **Input validation**: raise if both/neither of `input_ids` / `inputs_embeds` are given.
* **Capture-flags patch** at file end so the HF `generic.wrapper` knows which kwargs exist:

  ```python
  __all__ = ["LlamaForCausalLM","LlamaModel","LlamaPreTrainedModel"]

  try:
      _CAPTURE_FLAGS = ("output_attentions","output_hidden_states","return_dict","use_cache")
      LlamaModel.forward.capture_flags = _CAPTURE_FLAGS
      LlamaForCausalLM.forward.capture_flags = _CAPTURE_FLAGS
  except Exception:
      pass
  ```
* These prevent the “`NoneType` is not iterable” error in `transformers.utils.generic.wrapper`.

# What we did **not** change

* No layer sizes, head counts, rotary/MLP math, or parameter shapes.
* No change to inference speed/VRAM when `output_attentions=False`.
* LoRA/QLoRA compatibility is unchanged.

# Why this enables CBT

* Your **disentanglement loss** needs **per-head attention maps** each step.
* With the above, `outputs.attentions` gives a **list per layer** of tensors shaped `(batch, heads, seq, seq)`, so your loss can compute **pairwise head similarities** and penalize redundancy—pushing heads toward **specialized “construct” behaviors**.

That’s it: pure **observability + plumbing** changes to expose attentions end-to-end, with a tiny safety patch so HF’s trainer plays nicely.

# CBT Analysis

Utilities for analyzing TinyLlama + LoRA CBT training runs:
- Head entropy / concentration metrics
- Head ablations (Δloss)
- Heatmap exports
- Enterprise-oriented prompt probes

## Layout
- scripts/analyze_heads.py — main entry point
- data/ — place inputs or adapter weights if needed
- outputs/ — reports & heatmaps
