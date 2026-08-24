"""
Export the TinyCNNGRU model to ONNX and quantize to int8 for fast CPU
inference. This is the artifact you'd actually ship. (The Whisper-tiny model
can also be exported the same way via optimum, but TinyCNNGRU is the
intended low-latency production path.)

Run:
    python src/export_onnx.py --checkpoint checkpoints/tiny_best.pt \
        --out checkpoints/tiny_model.onnx --window_sec 2.5
"""
import argparse
import torch
from model import TinyCNNGRU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="checkpoints/tiny_model.onnx")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--window_sec", type=float, default=2.5)
    args = ap.parse_args()

    model = TinyCNNGRU(n_mels=args.n_mels)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    # T dimension from librosa mel settings in dataset.py: hop_length=160 -> ~100 frames/sec
    t_frames = int(args.window_sec * 16000 / 160) + 1
    dummy = torch.randn(1, args.n_mels, t_frames)

    torch.onnx.export(
        model, dummy, args.out,
        input_names=["log_mel"], output_names=["logit"],
        dynamic_axes={"log_mel": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported FP32 ONNX model to {args.out}")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quant_out = args.out.replace(".onnx", "_int8.onnx")
        quantize_dynamic(args.out, quant_out, weight_type=QuantType.QInt8)
        print(f"Exported quantized int8 model to {quant_out}")
        benchmark_onnx_models(args.out, quant_out, dummy.numpy())
    except ImportError:
        print("Install onnxruntime with quantization support to also produce the int8 version.")


def benchmark_onnx_models(fp32_path, int8_path, sample_input, n_runs=50):
    import os
    import time
    import onnxruntime as ort

    for label, path in [("FP32", fp32_path), ("INT8", int8_path)]:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        for _ in range(5):  # warmup
            sess.run(None, {input_name: sample_input})
        start = time.perf_counter()
        for _ in range(n_runs):
            sess.run(None, {input_name: sample_input})
        elapsed_ms = (time.perf_counter() - start) / n_runs * 1000
        size_kb = os.path.getsize(path) / 1024
        print(f"  {label}: {elapsed_ms:.2f} ms/inference, {size_kb:.1f} KB on disk")


if __name__ == "__main__":
    main()
