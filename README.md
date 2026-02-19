# microGPT GeLU MLA

A compact, pure-Python microGPT inspired by Andrej Karpathy's style, with two changes:
- `gelu` activation (instead of relu/relu^2)
- `multi-latent attention` slots (instead of multi-head attention)

It includes:
- scalar autograd engine (`Value`) with backprop
- GPT-style decoder-only transformer blocks
- first-principles Adam optimizer
- train loop + autoregressive text generation

## Run (Mac CPU friendly defaults)

```bash
python3 microgpt_mla.py
```

## Useful flags

```bash
python3 microgpt_mla.py \
  --steps 300 \
  --d_model 48 \
  --n_layer 2 \
  --n_latent 4 \
  --d_latent 12 \
  --d_ff 128 \
  --max_new_tokens 80 \
  --prompt "multi latent"
```

## Notes

- The script is intentionally small and readable, optimized for understanding.
- Training data is a tiny built-in corpus for quick local testing.
- Increasing `--steps` helps sample quality, but runtime also increases.

## Example run results

Training corpus used by the script:

```python
docs = [
    "the quick brown fox jumps over the lazy dog.",
    "small models teach big ideas with simple code.",
    "multi latent attention uses learned slots.",
    "gelu keeps gradients smoother than relu.",
]
```

Example command (training + inference on this corpus):

```bash
python3 microgpt_mla.py --steps 12 --print_every 2 --max_new_tokens 20 --temperature 0.6 --prompt "the "
```

Observed output:

```text
step    1 | loss 3.7038
step    2 | loss 3.7257
step    4 | loss 3.3858
step    6 | loss 3.3197
step    8 | loss 3.0639
step   10 | loss 3.0011
step   12 | loss 3.0347

--- sample ---
the qtm  tnunwnir ibbg e
```
