# src/marginkv/serving/client.py
"""Phase 8a: async client for vLLM's OpenAI-compatible server.

Token-id prompts, one generated token, yes/no logprobs (score = logprob(yes) - logprob(no)).
Prefix-cache hits come from usage.prompt_tokens_details.cached_tokens when the server reports it,
and always from the Prometheus counters vllm:prefix_cache_hits / vllm:prefix_cache_queries.
"""
import asyncio
import re
import time

import aiohttp
import numpy as np

YES_ID, NO_ID = 9693, 2152
_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?\s+(\S+)$")
_LABEL = re.compile(r'(\w+)="([^"]*)"')


def new_session(concurrency=16):
    conn = aiohttp.TCPConnector(limit=max(64, 2 * concurrency))
    return aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=None, sock_read=900))


async def wait_ready(session, base, timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            async with session.get(f"{base}/health") as r:
                if r.status == 200:
                    return time.time() - t0
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(2)
    raise TimeoutError(f"server at {base} not ready after {timeout_s} s")


async def get_text(session, url):
    async with session.get(url) as r:
        r.raise_for_status()
        return await r.text()


async def get_json(session, url):
    async with session.get(url) as r:
        r.raise_for_status()
        return await r.json()


def parse_metrics(text):
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, val = m.groups()
        try:
            v = float(val)
        except ValueError:
            continue
        rows.append((name, dict(_LABEL.findall(labels or "")), v))
    return rows


def prefix_counters(rows):
    out = {"queries": 0.0, "hits": 0.0}
    for name, _, v in rows:
        if name.endswith("_created"):
            continue
        if name.startswith("vllm:prefix_cache_queries"):
            out["queries"] += v
        elif name.startswith("vllm:prefix_cache_hits"):
            out["hits"] += v
    return out


def cache_config(rows):
    for name, labels, _ in rows:
        if name.startswith("vllm:cache_config_info"):
            return labels
    return {}


async def counters(session, base):
    return prefix_counters(parse_metrics(await get_text(session, f"{base}/metrics")))


async def server_info(session, base):
    rows = parse_metrics(await get_text(session, f"{base}/metrics"))
    try:
        version = (await get_json(session, f"{base}/version")).get("version")
    except aiohttp.ClientError:
        version = None
    m = (await get_json(session, f"{base}/v1/models"))["data"][0]
    return {"version": version, "model_id": m.get("id"), "model_root": m.get("root"),
            "max_model_len": m.get("max_model_len"), "cache_config": cache_config(rows)}


def check_setting(info, setting):
    """Abort if the running server is not the setting we are about to label results with."""
    root = str(info["model_root"])
    cc = info["cache_config"]
    want_root = "lora/df/merged" if setting.startswith("df") else "Qwen3-Reranker-0.6B"
    want_kv = "fp8" if setting == "df_fp8" else "auto"
    problems = []
    if want_root not in root:
        problems.append(f"model root {root!r} does not contain {want_root!r}")
    if cc and not str(cc.get("cache_dtype", "")).startswith(want_kv):
        problems.append(f"cache_dtype {cc.get('cache_dtype')!r}, expected {want_kv!r}")
    if cc and str(cc.get("enable_prefix_caching", "True")) != "True":
        problems.append(f"enable_prefix_caching = {cc.get('enable_prefix_caching')!r}")
    if problems:
        raise SystemExit(f"server does not match setting {setting}: " + "; ".join(problems))


async def reset_prefix_cache(session, base):
    async with session.post(f"{base}/reset_prefix_cache") as r:
        if r.status != 200:
            raise RuntimeError(f"/reset_prefix_cache returned {r.status} (VLLM_SERVER_DEV_MODE=1 set?)")
    await asyncio.sleep(0.2)


def _parse(body):
    ch = body["choices"][0]
    tops = (ch.get("logprobs") or {}).get("top_logprobs") or [None]
    top = tops[0] or {}
    usage = body.get("usage") or {}
    det = usage.get("prompt_tokens_details") or {}
    lp = lambda tid: float(top[f"token_id:{tid}"]) if f"token_id:{tid}" in top else None
    return {"lp_yes": lp(YES_ID), "lp_no": lp(NO_ID), "cached": det.get("cached_tokens"),
            "prompt_tokens": usage.get("prompt_tokens")}


async def complete(session, base, ids, model, logprobs, seed):
    payload = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 1.0, "seed": seed,
               "logprobs": logprobs, "allowed_token_ids": [NO_ID, YES_ID], "return_tokens_as_token_ids": True}
    t0 = time.perf_counter()
    async with session.post(f"{base}/v1/completions", json=payload) as r:
        status = r.status
        body = await r.json(content_type=None)
    lat = (time.perf_counter() - t0) * 1000.0
    if status != 200:
        return {"lp_yes": None, "lp_no": None, "cached": None, "prompt_tokens": None,
                "latency_ms": lat, "error": str(body)[:500]}
    out = _parse(body)
    out["latency_ms"] = lat
    out["error"] = None
    return out


async def run_requests(session, base, prompts, concurrency, model, logprobs, seed):
    """Closed loop: requests start in trace order, at most `concurrency` in flight."""
    sem = asyncio.Semaphore(concurrency)

    async def one(ids):
        async with sem:
            return await complete(session, base, ids, model, logprobs, seed)

    t0 = time.perf_counter()
    res = await asyncio.gather(*[one(p) for p in prompts])
    return list(res), time.perf_counter() - t0


def log_odds(r):
    return None if r["lp_yes"] is None or r["lp_no"] is None else r["lp_yes"] - r["lp_no"]


def pct(xs, p):
    return float(np.percentile(xs, p)) if len(xs) else None


def summarize(res, lengths, spans, wall_s):
    lat = [r["latency_ms"] for r in res]
    cached = [r["cached"] for r in res]
    have_cached = all(c is not None for c in cached)
    out = {"n": len(res), "wall_s": wall_s, "req_per_s": len(res) / wall_s,
           "prompt_tokens": int(sum(lengths)), "prompt_tok_per_s": sum(lengths) / wall_s,
           "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "mean": float(np.mean(lat)), "max": max(lat)},
           "errors": sum(r["error"] is not None for r in res),
           "missing_yes_no": sum(log_odds(r) is None for r in res),
           "per_request_cached_available": have_cached}
    if have_cached:
        doc_hit = sum(max(0, min(c, b) - a) for c, (a, b) in zip(cached, spans))
        doc_tot = sum(b - a for a, b in spans)
        out.update({"hit_tokens": int(sum(cached)), "hit_frac": sum(cached) / sum(lengths),
                    "doc_tokens_reused_frac": doc_hit / doc_tot,
                    "requests_with_hit_frac": sum(c > 0 for c in cached) / len(cached)})
    return out