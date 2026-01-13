"""Dynamic INT8 quantization of the exported ONNX model."""
import sys

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

SRC = sys.argv[1] if len(sys.argv) > 1 else "models/onnx"
DST = sys.argv[2] if len(sys.argv) > 2 else "models/onnx_int8"
TOK = sys.argv[3] if len(sys.argv) > 3 else "models/fp32"

model = ORTModelForSequenceClassification.from_pretrained(SRC)
quantizer = ORTQuantizer.from_pretrained(model)

# dynamic INT8, works on most x86 CPUs. Use .arm64(...) on ARM.
qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
quantizer.quantize(save_dir=DST, quantization_config=qconfig)

AutoTokenizer.from_pretrained(TOK).save_pretrained(DST)
print("quantized model saved to", DST)
