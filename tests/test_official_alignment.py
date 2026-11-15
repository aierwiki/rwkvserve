"""
rwkvserve RWKV-7 alignment/regression tests.

Numerical equivalence against the official RWKV-LM v7 implementation
(math copied verbatim into tests/ref_official.py), plus regressions for
specific fixes on the fix/official-rwkv7-alignment branch:
  D1 wkv7s state clone semantics (CPU/CUDA unified, no caller-buffer mutation)
  D2 init_state honors config.head_size_a
  D3 state-cache full-hit no longer double-processes the last prompt token
  D4 operator caches keyed by head_size

Run from repo root:  python3 tests/test_official_alignment.py
"""
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))          # tests/ref_official.py
sys.path.insert(0, str(ROOT))          # rwkvserve package

from ref_official import RWKV as OfficialRWKV, OfficialRNN  # noqa: E402

DT = torch.float32
PASS = True


def build_args(n_layer=3, n_embd=128, vocab=211, head_size=64):
    import types
    a = types.SimpleNamespace()
    a.n_layer, a.n_embd, a.vocab_size, a.head_size_a = n_layer, n_embd, vocab, head_size
    a.D_DECAY_LORA, a.D_AAA_LORA, a.D_MV_LORA, a.D_GATE_LORA = 16, 16, 8, 16
    return a


def seeded_official(args):
    torch.manual_seed(0)
    m = OfficialRWKV(args).to(DT).eval()
    # The official demo leaves raw nn.Parameter(torch.empty(...)) uninitialized
    # (it relies on load_state_dict). Init deterministically for testing, then
    # tame so the equivalence runs in a finite regime (untrained w0 can leave
    # decay~1 channels that blow the state up to NaN).
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith((".weight", ".bias")) and p.ndim >= 1:
                continue
            torch.nn.init.normal_(p, std=0.1)
        for n, p in m.named_parameters():
            if n.endswith("att.w0"):
                p.sub_(4.0)
            if n.endswith(".weight") and p.ndim == 2 and "emb" not in n and "head" not in n and "ln" not in n:
                p.mul_(0.5)
    return m


def build_serve_model(sd, args):
    from rwkvserve.models.rwkv7.model import RWKV7Model
    cfg = type("C", (), {})()
    cfg.n_layer, cfg.n_embd, cfg.vocab_size = args.n_layer, args.n_embd, args.vocab_size
    cfg.head_size_a, cfg.dim_att, cfg.ctx_len = args.head_size_a, args.n_embd, 4096
    cfg.grad_cp, cfg.use_peft = 0, False
    cfg.dim_att_lora, cfg.dim_gate_lora, cfg.dim_mv_lora = args.D_DECAY_LORA, args.D_GATE_LORA, args.D_MV_LORA
    torch.manual_seed(123)
    import re
    m = RWKV7Model(cfg).to(DT).eval()
    sd2 = {k: v for k, v in sd.items()
           if not (re.match(r"blocks\.0\.att\.v[012]$", k) and not hasattr(m.blocks[0].att, "v1"))}
    m.load_state_dict(sd2, strict=False)
    return m


def report(name, ok, detail=""):
    global PASS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    PASS = PASS and ok


def maxdiff(a, b):
    return (a.detach().float() - b.detach().float()).abs().max().item()


def main():
    args = build_args()
    ref = seeded_official(args)
    sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    serve = build_serve_model(sd, args)

    torch.manual_seed(7)
    idx = torch.randint(0, args.vocab_size, (2, 17))

    # T1 GPT-mode full-sequence logits vs official
    with torch.no_grad():
        l_ref, l_srv = ref(idx), serve(idx)
    report("T1 GPT-mode logits vs official", maxdiff(l_ref, l_srv) < 5e-4,
           f"diff={maxdiff(l_ref, l_srv):.2e}")

    # T2 token-by-token RNN vs official RNN loop
    rnn = OfficialRNN(ref, dtype=DT)
    tokens = idx[0].tolist()
    with torch.no_grad():
        st = rnn.init_state("cpu")
        ref_logits = []
        for t in tokens:
            out, st = rnn.forward(t, st)
            ref_logits.append(out)
        ref_logits = torch.stack(ref_logits)
        st3 = serve.init_state(1)
        srv_logits = []
        for t in tokens:
            lg, st3 = serve.forward_with_state(torch.tensor([[t]]), st3)
            srv_logits.append(lg[0, -1])
        srv_logits = torch.stack(srv_logits)
    report("T2 RNN token-by-token vs official RNN", maxdiff(ref_logits, srv_logits) < 5e-4,
           f"diff={maxdiff(ref_logits, srv_logits):.2e}")

    # T3 chunked prefill == full prefill
    with torch.no_grad():
        stc = serve.init_state(1)
        a_l, stc = serve.forward_with_state(idx[0, :5].unsqueeze(0), stc)
        b_l, stc = serve.forward_with_state(idx[0, 5:].unsqueeze(0), stc)
        d = maxdiff(serve(idx[0].unsqueeze(0))[0], torch.cat([a_l[0], b_l[0]], dim=0))
    report("T3 chunked prefill == full prefill", d < 5e-4, f"diff={d:.2e}")

    # T4 ragged batch
    with torch.no_grad():
        seqs = [idx[0, :9].tolist(), idx[1, :13].tolist()]
        outs, _ = serve.forward_batch_dynamic([list(s) for s in seqs], full_output=True)
        per = [serve(torch.tensor([s]))[0] for s in seqs]
        d = max(maxdiff(outs[i], per[i]) for i in range(2))
    report("T4 ragged batch forward", d < 5e-4, f"diff={d:.2e}")

    # T7 head_size != 64 (D2 + D4 regression)
    args32 = build_args(head_size=32)
    ref32 = seeded_official(args32)
    sd32 = {k: v.detach().clone() for k, v in ref32.state_dict().items()}
    serve32 = build_serve_model(sd32, args32)
    errs = []
    try:
        with torch.no_grad():
            st = serve32.init_state(1)
            H = args32.n_embd // args32.head_size_a
            expect = (1, H, args32.head_size_a, args32.head_size_a)
            if tuple(st[0]["att_kv"].shape) != expect:
                errs.append(f"init_state shape {tuple(st[0]['att_kv'].shape)} != {expect}")
            serve32.forward_with_state(torch.randint(0, args32.vocab_size, (1, 4)), st)
    except Exception as e:
        errs.append(f"EXC {type(e).__name__}: {e}")
    report("T7 head_size=32 state shape + forward", not errs, "; ".join(errs))

    # T8 scheduler full-cache-hit greedy consistency (D3 regression)
    import collections
    from rwkvserve.inference.scheduler_core import SchedulerCore, SequenceState

    class StubTok:
        idx2token = {i: bytes([i % 256]) for i in range(args.vocab_size)}

    class StubCache:
        def __init__(self):
            self.exact = None
        def lookup(self, tokens):
            if self.exact is not None and self.exact[0] == list(tokens):
                return len(tokens), self.exact[1], self.exact[2]
            return 0, None, None
        def create_checkpoints(self, tokens, messages, state, output):
            if self.exact is None:
                self.exact = (list(tokens), state, output)

    cache = StubCache()
    sched = SchedulerCore(serve, StubTok(), max_batch_size=4,
                          prefill_chunk_size=8, state_cache=cache)
    prompt = idx[0, :11].tolist()

    def run_req():
        seq = SequenceState(seq_id="r", prompt_tokens=list(prompt),
                            max_new_tokens=4, temperature=0.0, top_p=1.0)
        sched.reset()
        sched.add_sequence(seq)
        for _ in sched.run_to_completion_sync():
            pass
        return list(seq.generated_tokens)

    with torch.no_grad():
        g1 = run_req()
        g2 = run_req()
        stf = serve.init_state(1)
        lg, stf = serve.forward_with_state(torch.tensor([prompt]), stf)
        ref_g = []
        for _ in range(4):
            tok = int(lg[0, -1].float().argmax().item())
            ref_g.append(tok)
            lg, stf = serve.forward_with_state(torch.tensor([[tok]]), stf)
    report("T8 scheduler full-cache-hit greedy consistency", g1 == ref_g and g2 == ref_g,
           f"req1={g1} req2={g2} ref={ref_g}")

    print("\n===", "ALL PASS" if PASS else "FAILURES PRESENT", "===")
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
