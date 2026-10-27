"""Checks a candidate checkpoint against the baseline before any GPU is spent on it.

A new arm has to load through the shared notebook code unchanged, and it has to see the same
prompt. The notebooks render that prompt with each checkpoint's OWN chat template, so a
fine-tune that shipped an edited or missing template would silently change what the model
sees -- or crash `apply_chat_template` hours into a queue. Both are checkable from a few small
files on the Hub, with no weights downloaded:

  architecture, depth, width, vocab   must match the baseline (same loader, same tokenizer)
  weights                             safetensors only (no pickle, no remote code)
  chat template                       compared byte-for-byte against the baseline's

    python pipeline/scripts/preflight_arm.py Jianwen/Search-7B-SFT
    python pipeline/scripts/preflight_arm.py --report results/preflight.md <model> ...
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import date

BASELINE = "Qwen/Qwen2.5-7B-Instruct"
MATCH_KEYS = ["architectures", "model_type", "vocab_size", "hidden_size",
              "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
              "tie_word_embeddings", "rope_theta"]


def fetch(url, as_json=True):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            body = r.read().decode()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return json.loads(body) if as_json else body


def inspect(model):
    base = f"https://huggingface.co/{model}/resolve/main"
    info = fetch(f"https://huggingface.co/api/models/{model}")
    if info is None:
        raise SystemExit(f"{model}: not found on the Hub")
    files = [s["rfilename"] for s in info.get("siblings", [])]
    cfg = fetch(f"{base}/config.json")
    tok = fetch(f"{base}/tokenizer_config.json") or {}
    # Newer transformers saves the template as a separate file and leaves it out of
    # tokenizer_config.json; AutoTokenizer reads either.
    template = tok.get("chat_template")
    template_src = "tokenizer_config.json"
    if not template and "chat_template.jinja" in files:
        template = fetch(f"{base}/chat_template.jinja", as_json=False)
        template_src = "chat_template.jinja"
    return {
        "model": model,
        "gated": info.get("gated"),
        "params": (info.get("safetensors") or {}).get("parameters"),
        "safetensors": any(f.endswith(".safetensors") for f in files),
        "pickle": [f for f in files if f.endswith((".bin", ".pt", ".pth"))],
        "remote_code": any(f.endswith(".py") for f in files),
        "config": {k: cfg.get(k) for k in MATCH_KEYS},
        "template": template,
        "template_src": template_src if template else None,
        "eos": tok.get("eos_token"),
    }


def check(cand, base):
    problems = []
    for k in MATCH_KEYS:
        if cand["config"][k] != base["config"][k]:
            problems.append(f"config.{k}: {cand['config'][k]!r} != baseline {base['config'][k]!r}")
    if not cand["safetensors"]:
        problems.append("no safetensors weights")
    if cand["pickle"]:
        problems.append(f"pickle weights present: {cand['pickle']}")
    if cand["remote_code"]:
        problems.append("ships .py files (remote code)")
    if cand["gated"]:
        problems.append(f"gated: {cand['gated']}")
    if not cand["template"]:
        problems.append("no chat template -- apply_chat_template would fail")
    template_same = (cand["template"] or "").strip() == (base["template"] or "").strip()
    return problems, template_same


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--report", default=None, help="also write a markdown report here")
    args = ap.parse_args()

    base = inspect(BASELINE)
    lines = [f"# Preflight: candidate arms vs `{BASELINE}`", "",
             f"Generated {date.today()} by `pipeline/scripts/preflight_arm.py`. "
             f"Only configs and tokenizer files are read; no weights are downloaded.", "",
             "| model | params | arch/vocab/depth match | safetensors only | "
             "chat template vs baseline | verdict |",
             "|---|---|---|---|---|---|"]
    failed = False
    for m in args.models:
        cand = inspect(m)
        problems, same = check(cand, base)
        cfg_ok = not any(p.startswith("config.") for p in problems)
        verdict = "PASS" if not problems else "FAIL"
        failed |= bool(problems)
        tmpl = (f"byte-identical ({cand['template_src']})" if same
                else f"DIFFERS ({cand['template_src']})")
        lines.append(f"| `{m}` | {cand['params']} | {'yes' if cfg_ok else 'NO'} | "
                     f"{'yes' if cand['safetensors'] and not cand['pickle'] else 'NO'} | "
                     f"{tmpl} | {verdict} |")
        print(f"{m}: {verdict}  template {'identical' if same else 'DIFFERS'}")
        for p in problems:
            print(f"    {p}")
            lines.append(f"|  | ↳ {p} | | | | |")
    lines += ["", "A differing template is not automatically disqualifying (the RL arms render "
              "with their own templates too), but it must be read and recorded before the arm "
              "runs, because it changes the exact token sequence the model is shown."]
    if args.report:
        with open(args.report, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"-> {args.report}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
