| Training data | PyTorch FP32 acc % | ONNX INT8 acc % | INT8 change (points) | INT8 size MB |
|---|---|---|---|---|
| 8000 sentences (CPU) | 88.76 | 87.96 | -0.80 | 67.4 |
| 67,349 sentences (Kaggle GPU) | 90.71 | 90.83 | +0.12 | 67.4 |

Accuracy is measured on 872 validation sentences. The 95% margin of error is about +-2.1 points, so differences smaller than that are within the noise of this validation set. Latency is not compared here: it does not depend on the training data and it varies with the load of the machine.
