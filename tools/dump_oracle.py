"""Dev tool: dump ORT intermediate tensors as a per-node oracle.

Used while hand-writing the Zig forward passes (P2-P4): it exposes every node
output of an ONNX graph as a graph output (with optimizations DISABLED so node
names survive), runs onnxruntime on a representative input, and saves all
intermediate tensors to an .npz keyed by tensor name. Each Zig stage is then
checked against the corresponding tensor.

The dumps are large and regenerable, so they are git-ignored
(tools/oracle_out/). The committed regression oracle is the small public golden
corpus in tests/fixtures/ (see tools/capture_golden.py).

Run:
  uv --project python run python tools/dump_oracle.py            # all three
  uv --project python run python tools/dump_oracle.py encoder
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from hama.tokenizer import TextTokenizer
from hama.vocab import Vocabulary
from hama.asr import read_wav_mono

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets"  # shared source .onnx models
OUT = REPO / "tools" / "oracle_out"


def _expose_all(model_path: Path) -> str:
    """Return a temp ONNX path with every node output promoted to a graph output."""
    m = onnx.load(str(model_path))
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception:
        pass
    vis = {vi.name: vi for vi in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input)}
    existing = {o.name for o in m.graph.output}
    for node in m.graph.node:
        for o in node.output:
            if o and o not in existing:
                existing.add(o)
                vi = vis.get(o)
                if vi is None:
                    vi = onnx.ValueInfoProto()
                    vi.name = o
                m.graph.output.append(vi)
    tmp = Path(tempfile.mkdtemp()) / (model_path.stem + "_exposed.onnx")
    onnx.save(m, str(tmp))
    return str(tmp)


def _session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def _run_and_save(name: str, sess: ort.InferenceSession, feeds: dict) -> None:
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, feeds)
    arrays = {}
    for nm, arr in zip(out_names, outs):
        safe = nm.replace("/", "__").replace(":", "_")
        arrays[safe] = np.asarray(arr)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", **arrays)
    print(f"{name}: saved {len(arrays)} tensors -> {OUT / (name + '.npz')}")


def dump_encoder(text: str = "hello world") -> dict:
    vocab = Vocabulary.load()
    tok = TextTokenizer(vocab, max_input_len=128)
    enc = tok.encode(text)
    feeds = {
        "input_ids": enc.ids.reshape(1, -1).astype(np.int64),
        "input_lengths": np.array([enc.length], dtype=np.int64),
    }
    sess = _session(_expose_all(ASSETS / "encoder.onnx"))
    _run_and_save("encoder", sess, feeds)
    return feeds


def dump_decoder(text: str = "hello world") -> None:
    # Drive the real (unmodified) encoder + decoder loop to obtain step feeds,
    # then dump the decoder graph's intermediates for the first two steps.
    vocab = Vocabulary.load()
    tok = TextTokenizer(vocab, max_input_len=128)
    enc = tok.encode(text)
    enc_sess = ort.InferenceSession(str(ASSETS / "encoder.onnx"), providers=["CPUExecutionProvider"])
    eo, pk, hidden, emask, prev = enc_sess.run(
        ["encoder_outputs", "projected_keys", "hidden", "encoder_mask", "prev_attn"],
        {"input_ids": enc.ids.reshape(1, -1).astype(np.int64), "input_lengths": np.array([enc.length], dtype=np.int64)},
    )
    src_len = eo.shape[1]
    positions = np.arange(src_len, dtype=np.float32).reshape(1, src_len)
    dec_path = _expose_all(ASSETS / "decoder_step.onnx")
    dec_sess = _session(dec_path)
    dec_in = np.array([[vocab.sos_id]], dtype=np.int64)
    for step in range(2):
        feeds = {
            "decoder_input_ids": dec_in,
            "encoder_outputs": eo,
            "projected_keys": pk,
            "encoder_mask": emask,
            "prev_attn": prev,
            "hidden": hidden,
            "positions": positions,
        }
        _run_and_save(f"decoder_step{step}", dec_sess, feeds)
        # advance using clean (named) outputs
        nt, hidden, prev, _ = dec_sess.run(
            ["next_token_ids", "hidden_out", "prev_attn_out", "attn_argmax"], feeds
        )
        dec_in = np.array([[int(np.asarray(nt).reshape(-1)[0])]], dtype=np.int64)


def dump_asr(wav: str = "tone_1000_2s_16k.wav") -> None:
    waveform, sr = read_wav_mono(REPO / "tests" / "fixtures" / "audio" / wav)
    feeds = {
        "waveform": waveform.reshape(1, -1).astype(np.float32),
        "waveform_lengths": np.array([waveform.shape[0]], dtype=np.int64),
    }
    sess = _session(_expose_all(ASSETS / "asr_waveform_fp16.onnx"))
    _run_and_save("asr", sess, feeds)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "encoder"):
        dump_encoder()
    if which in ("all", "decoder"):
        dump_decoder()
    if which in ("all", "asr"):
        dump_asr()


if __name__ == "__main__":
    main()
