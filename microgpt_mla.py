import argparse
import math
import random


# autograd
class Value:
    def __init__(self, data, prev=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(prev)
        self._op = op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        out._backward = lambda: (setattr(self, "grad", self.grad + out.grad), setattr(other, "grad", other.grad + out.grad))
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        out._backward = lambda: (setattr(self, "grad", self.grad + other.data * out.grad), setattr(other, "grad", other.grad + self.data * out.grad))
        return out

    def __pow__(self, p):
        out = Value(self.data**p, (self,), f"**{p}")
        out._backward = lambda: setattr(self, "grad", self.grad + (p * self.data ** (p - 1)) * out.grad)
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        out._backward = lambda: setattr(self, "grad", self.grad + e * out.grad)
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        out._backward = lambda: setattr(self, "grad", self.grad + (1.0 / self.data) * out.grad)
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        out._backward = lambda: setattr(self, "grad", self.grad + (1 - t * t) * out.grad)
        return out

    def backward(self):
        topo, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other**-1)
    def __rtruediv__(self, other): return other * (self**-1)


def gelu(x):
    # gelu
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1 + (c * (x + 0.044715 * (x**3))).tanh())


def softmax(xs):
    # softmax
    m = max(x.data for x in xs)
    exps = [(x - m).exp() for x in xs]
    z = sum(exps, Value(0.0))
    return [e / z for e in exps]


def init_vec(n, scale=1.0):
    return [Value(random.gauss(0.0, scale)) for _ in range(n)]


def ones_vec(n):
    return [Value(1.0) for _ in range(n)]


def init_mat(r, c, scale):
    return [init_vec(c, scale) for _ in range(r)]


def matvec(x, w):
    y = []
    for row in w:
        s = Value(0.0)
        for wi, xi in zip(row, x):
            s += wi * xi
        y.append(s)
    return y


def rmsnorm(x, g, eps=1e-6):
    ms = sum((xi * xi for xi in x), Value(0.0)) / len(x)
    inv = (ms + eps) ** -0.5
    return [xi * inv * gi for xi, gi in zip(x, g)]


def split_lat(v, n_latent, d_latent):
    return [v[i * d_latent:(i + 1) * d_latent] for i in range(n_latent)]


def init_model(cfg, vocab):
    d, nl, nlat, dlat, dff = cfg["d_model"], cfg["n_layer"], cfg["n_latent"], cfg["d_latent"], cfg["d_ff"]
    s = 1.0 / math.sqrt(d)
    sd = {
        "wte": [init_vec(d, s) for _ in range(vocab)],
        "wpe": [init_vec(d, s) for _ in range(cfg["block_size"])],
        "ln_f": ones_vec(d),
        "lm_head": init_mat(vocab, d, s),
    }
    for i in range(nl):
        p = f"layer{i}."
        sd[p + "ln1"] = ones_vec(d)
        sd[p + "wq"] = init_mat(nlat * dlat, d, s)
        sd[p + "wk"] = init_mat(nlat * dlat, d, s)
        sd[p + "wv"] = init_mat(nlat * dlat, d, s)
        sd[p + "wo"] = init_mat(d, nlat * dlat, s)
        sd[p + "ln2"] = ones_vec(d)
        sd[p + "fc1"] = init_mat(dff, d, s)
        sd[p + "fc2"] = init_mat(d, dff, s)
    return sd


def all_params(sd):
    ps, st = [], list(sd.values())
    while st:
        x = st.pop()
        if isinstance(x, Value): ps.append(x)
        elif isinstance(x, list): st.extend(x)
    return ps


def gpt_token(token_id, pos_id, cache, sd, cfg):
    x = [a + b for a, b in zip(sd["wte"][token_id], sd["wpe"][pos_id])]
    for li in range(cfg["n_layer"]):
        p = f"layer{li}."

        # multi-latent attention
        res = x
        x = rmsnorm(x, sd[p + "ln1"])
        q = split_lat(matvec(x, sd[p + "wq"]), cfg["n_latent"], cfg["d_latent"])
        k = split_lat(matvec(x, sd[p + "wk"]), cfg["n_latent"], cfg["d_latent"])
        v = split_lat(matvec(x, sd[p + "wv"]), cfg["n_latent"], cfg["d_latent"])
        cache[li]["k"].append(k)
        cache[li]["v"].append(v)

        mixed = []
        for l in range(cfg["n_latent"]):
            logits = []
            for t in range(len(cache[li]["k"])):
                dot = sum((q[l][j] * cache[li]["k"][t][l][j] for j in range(cfg["d_latent"])), Value(0.0))
                logits.append(dot / math.sqrt(cfg["d_latent"]))
            w = softmax(logits)
            for j in range(cfg["d_latent"]):
                s = Value(0.0)
                for t, wt in enumerate(w):
                    s += wt * cache[li]["v"][t][l][j]
                mixed.append(s)

        x = [a + b for a, b in zip(matvec(mixed, sd[p + "wo"]), res)]

        # mlp + gelu
        res = x
        x = rmsnorm(x, sd[p + "ln2"])
        h = [gelu(z) for z in matvec(x, sd[p + "fc1"])]
        x = [a + b for a, b in zip(matvec(h, sd[p + "fc2"]), res)]

    return matvec(rmsnorm(x, sd["ln_f"]), sd["lm_head"])


def adam(params, m, v, step, lr, b1=0.9, b2=0.95, eps=1e-8):
    for i, p in enumerate(params):
        g = p.grad
        m[i] = b1 * m[i] + (1 - b1) * g
        v[i] = b2 * v[i] + (1 - b2) * g * g
        mh = m[i] / (1 - b1**step)
        vh = v[i] / (1 - b2**step)
        p.data -= lr * mh / (math.sqrt(vh) + eps)
        p.grad = 0.0


def sample_id(probs):
    r, c = random.random(), 0.0
    for i, p in enumerate(probs):
        c += p.data
        if r <= c: return i
    return len(probs) - 1


def train_and_generate(cfg):
    docs = [
        "the quick brown fox jumps over the lazy dog.",
        "small models teach big ideas with simple code.",
        "multi latent attention uses learned slots.",
        "gelu keeps gradients smoother than relu.",
    ]
    corpus = "\n".join(docs)
    chars = sorted(set(corpus + cfg["prompt"]))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    corpus_ids = [stoi[c] for c in corpus]

    sd = init_model(cfg, len(chars))
    params = all_params(sd)
    m, v = [0.0] * len(params), [0.0] * len(params)

    for step in range(1, cfg["steps"] + 1):
        n = min(cfg["train_seq_len"], cfg["block_size"] - 1, len(corpus_ids) - 1)
        if n <= 0: continue
        st = random.randint(0, len(corpus_ids) - n - 1)
        toks = corpus_ids[st:st + n + 1]

        cache = [{"k": [], "v": []} for _ in range(cfg["n_layer"])]
        losses = []
        for pos in range(n):
            probs = softmax(gpt_token(toks[pos], pos, cache, sd, cfg))
            losses.append(-(probs[toks[pos + 1]].log()))

        loss = sum(losses, Value(0.0)) / n
        loss.backward()
        adam(params, m, v, step, cfg["lr"])
        if step == 1 or step % cfg["print_every"] == 0:
            print(f"step {step:4d} | loss {loss.data:.4f}")

    # inference
    ids = [stoi.get(c, 0) for c in cfg["prompt"]][:cfg["block_size"] - 1]
    out = list(ids)
    cache = [{"k": [], "v": []} for _ in range(cfg["n_layer"])]
    logits = None
    for pos, tid in enumerate(ids):
        logits = gpt_token(tid, pos, cache, sd, cfg)
    if logits is None:
        logits = gpt_token(0, 0, cache, sd, cfg)

    for _ in range(cfg["max_new_tokens"]):
        probs = softmax([z / max(cfg["temperature"], 1e-6) for z in logits])
        nxt = sample_id(probs)
        if len(out) >= cfg["block_size"] - 1: break
        out.append(nxt)
        logits = gpt_token(nxt, len(out) - 1, cache, sd, cfg)

    print("\n--- sample ---")
    print("".join(itos[i] for i in out))


def main():
    ap = argparse.ArgumentParser(description="microGPT with multi-latent attention + gelu")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--train_seq_len", type=int, default=20)
    ap.add_argument("--d_model", type=int, default=24)
    ap.add_argument("--n_layer", type=int, default=1)
    ap.add_argument("--n_latent", type=int, default=2)
    ap.add_argument("--d_latent", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=48)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--prompt", type=str, default="multi latent")
    cfg = vars(ap.parse_args())
    random.seed(42)
    train_and_generate(cfg)


if __name__ == "__main__":
    main()
