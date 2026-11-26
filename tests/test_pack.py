# tests/test_pack.py
"""Phase 5.5 packing tests (CPU, synthetic KV, seconds).
Nibble round trip; for every format and doc lengths 257 / 40 / 5: tensor specs as predicted, file size
== predicted_file_bytes, payload == theory_bytes, unpack == quantize_doc bit for bit; shared file
round trip exact."""
import os
import tempfile

import torch

from marginkv.kv import pack as P
from marginkv.kv import quant as Q
from marginkv.kv.cache_ops import DocKV

SHARED, L = 59, 3


def synth_kv(n_doc, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = SHARED + n_doc
    keys, values = [], []
    for _ in range(L):
        k = torch.randn(8, n, 128, generator=g) * 2
        k[:, :, :4] *= 40
        keys.append(k.to(torch.bfloat16))
        values.append(torch.randn(8, n, 128, generator=g).to(torch.bfloat16))
    return DocKV(keys, values, next_pos=n, shared_len=SHARED)


def test_nibbles():
    c = torch.randint(0, 16, (3, 8, 37, 128), dtype=torch.uint8)
    b = P.pack_nibbles(c)
    assert b.shape == (3, 8, 37, 64) and torch.equal(P.unpack_nibbles(b), c)
    print("nibble pack/unpack: PASS")


def test_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        for n_doc in (257, 40, 5):
            kv = synth_kv(n_doc, seed=n_doc)
            own_k = [k[:, :SHARED] for k in kv.keys]
            own_v = [v[:, :SHARED] for v in kv.values]
            for fmt in Q.FORMATS:
                path = os.path.join(tmp, f"{fmt}_{n_doc}.safetensors")
                info = P.pack_doc(kv, fmt, path)
                want_specs = P.tensor_specs(fmt, n_doc, layers=L)
                assert info["specs"] == want_specs, (fmt, n_doc, info["specs"], want_specs)
                assert info["file_bytes"] == P.predicted_file_bytes(fmt, n_doc, SHARED, layers=L), (fmt, n_doc)
                assert info["payload_bytes"] == Q.theory_bytes(fmt, n_doc, layers=L), (fmt, n_doc)
                u = P.unpack_doc(path, own_k, own_v)
                ref = Q.quantize_doc(kv, fmt)
                assert u.next_pos == ref.next_pos and u.shared_len == ref.shared_len
                assert all(torch.equal(a, b) for a, b in zip(u.keys + u.values, ref.keys + ref.values)), (fmt, n_doc)
        shared = DocKV(own_k, own_v, next_pos=SHARED, shared_len=SHARED)
        path = os.path.join(tmp, "shared.safetensors")
        P.save_shared(shared, path)
        k, v, meta = P.load_shared(path)
        assert meta["shared_len"] == str(SHARED)
        assert all(torch.equal(a, b) for a, b in zip(k + v, own_k + own_v))
    print("pack/unpack bit-exact, sizes == predicted, payload == theory, shared round trip: PASS")


if __name__ == "__main__":
    test_nibbles()
    test_roundtrip()