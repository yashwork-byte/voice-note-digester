---
license: mit
base_model: ai4bharat/indic-conformer-600m-multilingual
language: [hi, bn, ta, as, gu, kn, ml, mr, or, pa, te, ur]
tags: [asr, speech-recognition, onnx, int8, quantized, indic]
---

# indic-conformer-600m-int8-onnx

Int8 dynamically-quantized copy of
[ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
(MIT): the ONNX encoder is quantized from **2.4 GB fp32 (external data) to a
self-contained 0.62 GB int8 file**; everything else (CTC decoder,
per-language heads, preprocessor, remote code) is unchanged.

Why: the fp32 encoder needs ~2.6 GB resident — too much next to an LLM on an
8 GB device. Int8 cuts peak RSS to ~1.1 GB and speeds up loading ~3×, with a
measured WER cost of ≈ +0.5 points on one of six spot-checked conversational
clips (the rest bit-identical).

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("yashwork-byte/indic-conformer-600m-int8-onnx",
                                  trust_remote_code=True)
text = model(wav_16k_mono_tensor, "hi", "ctc")
```

Built for the [voice-note-digester](https://github.com/yashwork-byte/voice-note-digester)
project (`quantize_stt.py` — onnxruntime dynamic quantization, QInt8
weights). All credit for the model to [AI4Bharat](https://ai4bharat.iitm.ac.in/);
this repo only repackages it for memory-constrained on-device use.
