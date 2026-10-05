import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

device = "cpu"
VOCAB_SIZE = 8192
BLOCK = 256

tok = Tokenizer.from_file("pico_tokenizer.json")
enc = lambda s: tok.encode(s).ids
dec = lambda ids: tok.decode(ids, skip_special_tokens=True)
BOS = tok.token_to_id("<bos>")
EOS = tok.token_to_id("<eos>")
USER = tok.token_to_id("<user>")
PICO = tok.token_to_id("<pico>")

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block, block)).view(1, 1, block, block))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, block)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class PicoGPT(nn.Module):
    def __init__(self, vocab_size, block, n_layer=6, n_head=6, n_embd=384):
        super().__init__()
        self.block = block
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.3, top_k=10):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            if (next_tok == EOS).all():
                break
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

model = PicoGPT(VOCAB_SIZE, BLOCK).to(device)
model.load_state_dict(torch.load("pico-15m.pt", map_location=device))
model.eval()

print("--- Pico 15M General Assistant (15M-AI) ---")
print("Escribe 'exit' para salir.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["salir", "exit", "quit"]:
        break
    
    prompt = f"<user>{user_input}<pico>"
    ids = enc(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    
    out = model.generate(idx, temperature=0.3, top_k=10)[0].tolist()
    
    response_ids = out[len(ids):]
    response = dec(response_ids).strip()

    print(f"Pico: {response}\n")
