"""Build Zig validation fixtures (.hama archives of ground-truth tensors) from
the optimization-disabled ORT oracle, so `zig build test` can validate each
hand-written model stage against ONNX Runtime.

Fixtures are committed under zig/src/models/fixtures/ and embedded by the Zig
model tests.

Run:
  uv --project python run python tools/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hama.tokenizer import TextTokenizer
from hama.vocab import Vocabulary

from dump_oracle import _expose_all, _session, ASSETS  # noqa: E402
from hama_pack import pack  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "zig" / "src" / "models" / "fixtures"


def encoder_fixture(text: str = "hello world") -> None:
    vocab = Vocabulary.load()
    tok = TextTokenizer(vocab, max_input_len=128)
    enc = tok.encode(text)
    ids = enc.ids.reshape(1, -1).astype(np.int64)
    feeds = {"input_ids": ids, "input_lengths": np.array([enc.length], dtype=np.int64)}
    sess = _session(_expose_all(ASSETS / "encoder.onnx"))

    want = {
        "ln1": "/frontend/norm/LayerNormalization_output_0",
        "frontend_add": "/frontend/Add_output_0",
        "gru1": "/encoder/Reshape_output_0",
        "gru2": "/encoder/Reshape_1_output_0",
        "encoder_outputs": "encoder_outputs",
        "projected_keys": "projected_keys",
        "hidden": "hidden",
        "encoder_mask": "encoder_mask",
        "prev_attn": "prev_attn",
    }
    outs = sess.run(list(want.values()), feeds)
    by_name = dict(zip(want.keys(), outs))

    tensors: dict[str, np.ndarray] = {
        "input_ids": enc.ids.astype(np.int64),
        "length": np.array([enc.length], dtype=np.int64),
        "ln1": np.ascontiguousarray(by_name["ln1"].reshape(128, 80).astype(np.float32)),
        "frontend_add": np.ascontiguousarray(by_name["frontend_add"].reshape(128, 80).astype(np.float32)),
        "gru1": np.ascontiguousarray(by_name["gru1"].reshape(128, 192).astype(np.float32)),
        "gru2": np.ascontiguousarray(by_name["gru2"].reshape(128, 192).astype(np.float32)),
        "encoder_outputs": np.ascontiguousarray(by_name["encoder_outputs"].reshape(128, 192).astype(np.float32)),
        "projected_keys": np.ascontiguousarray(by_name["projected_keys"].reshape(128, 96).astype(np.float32)),
        "hidden": np.ascontiguousarray(by_name["hidden"].reshape(2, 96).astype(np.float32)),
        "encoder_mask": np.ascontiguousarray(by_name["encoder_mask"].reshape(128).astype(np.uint8)),
        "prev_attn": np.ascontiguousarray(by_name["prev_attn"].reshape(128).astype(np.float32)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pack(OUT / "encoder_hello.hama", tensors)
    print(f"encoder fixture -> {OUT / 'encoder_hello.hama'}  (len={enc.length})")


def decoder_fixture(text: str = "hello world") -> None:
    import onnxruntime as ort

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

    dec_sess = _session(_expose_all(ASSETS / "decoder_step.onnx"))
    want = {
        "state_norm": "/decoder_state_norm/LayerNormalization_output_0",
        "attn": "/Softmax_output_0",
        "context_norm": "/context_norm/LayerNormalization_output_0",
        "logits": "/output_proj/MatMul_output_0",
        "next_token_ids": "next_token_ids",
        "hidden_out": "hidden_out",
        "prev_attn_out": "prev_attn_out",
        "attn_argmax": "attn_argmax",
    }

    dec_in = np.array([[vocab.sos_id]], dtype=np.int64)
    cur_hidden, cur_prev = hidden, prev
    # Capture two steps: step 0 (uniform prev_attn) and step 1 (real attention).
    for step in range(2):
        feeds = {
            "decoder_input_ids": dec_in,
            "encoder_outputs": eo,
            "projected_keys": pk,
            "encoder_mask": emask,
            "prev_attn": cur_prev,
            "hidden": cur_hidden,
            "positions": positions,
        }
        outs = dict(zip(want.keys(), dec_sess.run(list(want.values()), feeds)))
        tensors: dict[str, np.ndarray] = {
            "decoder_input_ids": np.ascontiguousarray(dec_in.reshape(1).astype(np.int64)),
            "encoder_outputs": np.ascontiguousarray(eo.reshape(src_len, 192).astype(np.float32)),
            "projected_keys": np.ascontiguousarray(pk.reshape(src_len, 96).astype(np.float32)),
            "encoder_mask": np.ascontiguousarray(emask.reshape(src_len).astype(np.uint8)),
            "prev_attn": np.ascontiguousarray(cur_prev.reshape(src_len).astype(np.float32)),
            "hidden": np.ascontiguousarray(cur_hidden.reshape(2, 96).astype(np.float32)),
            "positions": np.ascontiguousarray(positions.reshape(src_len).astype(np.float32)),
            "state_norm": np.ascontiguousarray(outs["state_norm"].reshape(96).astype(np.float32)),
            "attn": np.ascontiguousarray(outs["attn"].reshape(src_len).astype(np.float32)),
            "context_norm": np.ascontiguousarray(outs["context_norm"].reshape(192).astype(np.float32)),
            "logits": np.ascontiguousarray(outs["logits"].reshape(189).astype(np.float32)),
            "next_token_id": np.array([int(np.asarray(outs["next_token_ids"]).reshape(-1)[0])], dtype=np.int64),
            "hidden_out": np.ascontiguousarray(outs["hidden_out"].reshape(2, 96).astype(np.float32)),
            "prev_attn_out": np.ascontiguousarray(outs["prev_attn_out"].reshape(src_len).astype(np.float32)),
            "attn_argmax": np.array([int(np.asarray(outs["attn_argmax"]).reshape(-1)[0])], dtype=np.int64),
        }
        pack(OUT / f"decoder_step{step}.hama", tensors)
        print(f"decoder fixture step{step} -> next_token={tensors['next_token_id'][0]} attn_argmax={tensors['attn_argmax'][0]}")
        # advance with clean named outputs
        nt, cur_hidden, cur_prev, _ = dec_sess.run(
            ["next_token_ids", "hidden_out", "prev_attn_out", "attn_argmax"], feeds
        )
        dec_in = np.array([[int(np.asarray(nt).reshape(-1)[0])]], dtype=np.int64)


def g2p_e2e_fixture(text: str = "hello world", max_output_len: int = 32) -> None:
    """Drive the full ORT encoder + greedy decoder loop and save the decoded
    token / attention sequence, so a single Zig test validates the whole G2P
    pipeline (encoder + loop + EOS handling)."""
    import onnxruntime as ort

    vocab = Vocabulary.load()
    tok = TextTokenizer(vocab, max_input_len=128)
    enc = tok.encode(text)
    enc_sess = ort.InferenceSession(str(ASSETS / "encoder.onnx"), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(ASSETS / "decoder_step.onnx"), providers=["CPUExecutionProvider"])
    eo, pk, hidden, emask, prev = enc_sess.run(
        ["encoder_outputs", "projected_keys", "hidden", "encoder_mask", "prev_attn"],
        {"input_ids": enc.ids.reshape(1, -1).astype(np.int64), "input_lengths": np.array([enc.length], dtype=np.int64)},
    )
    src_len = eo.shape[1]
    positions = np.arange(src_len, dtype=np.float32).reshape(1, src_len)
    dec_in = np.array([[vocab.sos_id]], dtype=np.int64)
    tokens, attns = [], []
    for _ in range(max_output_len):
        nt, hidden, prev, aa = dec_sess.run(
            ["next_token_ids", "hidden_out", "prev_attn_out", "attn_argmax"],
            {
                "decoder_input_ids": dec_in,
                "encoder_outputs": eo,
                "projected_keys": pk,
                "encoder_mask": emask,
                "prev_attn": prev,
                "hidden": hidden,
                "positions": positions,
            },
        )
        token = int(np.asarray(nt).reshape(-1)[0])
        tokens.append(token)
        attns.append(int(np.asarray(aa).reshape(-1)[0]))
        dec_in = np.array([[token]], dtype=np.int64)
        if token == vocab.eos_id:
            break

    tensors = {
        "input_ids": enc.ids.astype(np.int64),
        "length": np.array([enc.length], dtype=np.int64),
        "sos_id": np.array([vocab.sos_id], dtype=np.int64),
        "eos_id": np.array([vocab.eos_id], dtype=np.int64),
        "tokens": np.array(tokens, dtype=np.int64),
        "attns": np.array(attns, dtype=np.int64),
    }
    pack(OUT / "g2p_hello.hama", tensors)
    print(f"g2p e2e fixture -> {len(tokens)} tokens: {tokens}")


def asr_fixture(wav: str = "short_100ms_16k.wav") -> None:
    from hama.asr import read_wav_mono

    waveform, sr = read_wav_mono(REPO / "tests" / "fixtures" / "audio" / wav)
    feeds = {
        "waveform": waveform.reshape(1, -1).astype(np.float32),
        "waveform_lengths": np.array([waveform.shape[0]], dtype=np.int64),
    }
    sess = _session(_expose_all(ASSETS / "asr_waveform_fp16.onnx"))
    want = {
        "stft_real": "/Conv_output_0",
        "logmel": "/Log_output_0",
        "stem": "/acoustic_model/stem/stem.2/Mul_1_output_0",
        "attn1": "/acoustic_model/export_attn_layers.1/Add_4_output_0",
        "log_probs": "log_probs",
        "out_lengths": "out_lengths",
    }
    for i in range(11):
        want[f"backbone{i}"] = f"/acoustic_model/backbone/backbone.{i}/Add_output_0"
    outs = dict(zip(want.keys(), sess.run(list(want.values()), feeds)))
    T_stft = outs["stft_real"].shape[2]
    T = int(outs["out_lengths"].reshape(-1)[0])
    tensors = {
        "waveform": np.ascontiguousarray(waveform.astype(np.float32)),
        "waveform_length": np.array([waveform.shape[0]], dtype=np.int64),
        "stft_real": np.ascontiguousarray(outs["stft_real"].reshape(201, T_stft).astype(np.float32)),
        "logmel": np.ascontiguousarray(outs["logmel"].reshape(T_stft, 80).astype(np.float32)),
        "stem": np.ascontiguousarray(outs["stem"].reshape(256, T).astype(np.float32)),
        "attn1": np.ascontiguousarray(outs["attn1"].reshape(T, 256).astype(np.float32)),
        "log_probs": np.ascontiguousarray(outs["log_probs"].reshape(T, 191).astype(np.float32)),
        "out_length": np.array([T], dtype=np.int64),
    }
    for i in range(11):
        tensors[f"backbone{i}"] = np.ascontiguousarray(outs[f"backbone{i}"].reshape(256, T).astype(np.float32))
    pack(OUT / "asr_short.hama", tensors)
    print(f"asr fixture -> T_stft={T_stft} T={T}  (wav={wav}, N={waveform.shape[0]})")


def main() -> None:
    encoder_fixture()
    decoder_fixture()
    g2p_e2e_fixture()
    asr_fixture()


if __name__ == "__main__":
    main()
