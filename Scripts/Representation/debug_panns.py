from pathlib import Path
import numpy as np
import torch

from panns_adapter import PANNsAdapter
from window_dataset import WindowDataset

# ========= 改这里 =========
CHECKPOINT_PATH = r"D:\PycharmProjects\HeartSoundPhenotype\Representation_learning\panns\Cnn14_mAP=0.431.pth"
WINDOW_INDEX_CSV = r"D:\PycharmProjects\HeartSoundPhenotype\Data_preprocessing\Data_representation\window_index.csv"
RECORDING_MANIFEST_CSV = r"D:\PycharmProjects\HeartSoundPhenotype\Data_preprocessing\Data_representation\recording_manifest.csv"
DEVICE = "cuda"   # 没GPU就改成 "cpu"
SAMPLE_INDEX = 0  # 先随便取第几个窗口做测试
# =========================

def describe_output(output, prefix="output"):
    if torch.is_tensor(output):
        print(f"{prefix}: Tensor, shape={tuple(output.shape)}, dtype={output.dtype}")
        return

    if isinstance(output, dict):
        print(f"{prefix}: dict")
        print("keys =", list(output.keys()))
        for k, v in output.items():
            if torch.is_tensor(v):
                print(f"  - {k}: Tensor, shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  - {k}: {type(v)}")
        return

    if isinstance(output, (list, tuple)):
        print(f"{prefix}: {type(output).__name__}, len={len(output)}")
        for i, v in enumerate(output):
            if torch.is_tensor(v):
                print(f"  - [{i}]: Tensor, shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  - [{i}]: {type(v)}")
        return

    print(f"{prefix}: unsupported type = {type(output)}")


def main():
    # 1) 先加载一个窗口，确认输入没问题
    dataset = WindowDataset(
        window_index_csv=WINDOW_INDEX_CSV,
        recording_manifest_csv=RECORDING_MANIFEST_CSV,
        patient_ids=None,
    )
    item = dataset[SAMPLE_INDEX]
    waveform = item["waveform"]  # 1D float32, src_sr 固定 8000
    src_sr = int(item["src_sr"])

    print("Loaded one window:")
    print("  patient_id =", item["patient_id"])
    print("  position   =", item["position"])
    print("  window_id  =", item["window_id"])
    print("  waveform shape =", waveform.shape, "dtype =", waveform.dtype)
    print("  src_sr =", src_sr)

    # 2) 加载你当前这版 PANNsAdapter
    adapter = PANNsAdapter(
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
    )

    # 3) 走 adapter 的预处理：8k -> 32k + right pad
    batch_waveforms = adapter._prepare_batch([waveform], src_sr=src_sr)
    print("\nAfter adapter._prepare_batch():")
    print("  batch_waveforms shape =", tuple(batch_waveforms.shape))
    print("  batch_waveforms dtype =", batch_waveforms.dtype)
    print("  device =", batch_waveforms.device)

    # 4) 直接看模型原始前向输出
    with torch.no_grad():
        raw_output = adapter.model(batch_waveforms)

    print("\n=== Raw model output description ===")
    describe_output(raw_output, prefix="raw_output")

    # 5) 明确判断当前 checkpoint 是否有 embedding
    print("\n=== Embedding availability check ===")
    if isinstance(raw_output, dict):
        if "embedding" in raw_output:
            emb = raw_output["embedding"]
            print("Found key: 'embedding'")
            print("embedding shape =", tuple(emb.shape))
            print("=> 这个 checkpoint 在你当前 adapter 下可以直接取 embedding。")
        elif "clipwise_output" in raw_output:
            print("Found key: 'clipwise_output' but NO 'embedding'")
            print("clipwise_output shape =", tuple(raw_output["clipwise_output"].shape))
            print("=> 这个 checkpoint 在你当前 adapter 下不能直接取 embedding，会触发你代码里的报错。")
        else:
            print("No 'embedding' key. Available keys =", list(raw_output.keys()))
            print("=> 需要你手动决定取哪个中间输出，或者改 adapter。")
    elif torch.is_tensor(raw_output):
        print("Model output is a tensor directly, shape =", tuple(raw_output.shape))
        print("=> 这种情况未必是 embedding，也可能是 logits，需要你确认模型实现。")
    else:
        print("Unsupported raw output type:", type(raw_output))

    # 6) 顺手再测试一下你当前 adapter 的 extract_embeddings 能不能跑通
    print("\n=== Adapter.extract_embeddings() smoke test ===")
    try:
        emb_np = adapter.extract_embeddings([waveform], src_sr=src_sr)
        print("adapter.extract_embeddings() succeeded.")
        print("returned shape =", emb_np.shape, "dtype =", emb_np.dtype)
    except Exception as e:
        print("adapter.extract_embeddings() failed:")
        print(repr(e))


if __name__ == "__main__":
    main()