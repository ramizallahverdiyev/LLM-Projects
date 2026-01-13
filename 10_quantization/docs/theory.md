# Model Quantization Theory

This document provides a high-level overview of model quantization, aimed at data scientists and ML practitioners who are new to the topic.

## What is Quantization?

At its core, **quantization** is the process of reducing the precision of the numbers used to represent a model's parameters (weights and biases). In deep learning, these parameters are typically stored as 32-bit floating-point numbers (FP32). Quantization converts these numbers into lower-precision formats, such as 16-bit floating-point (FP16), 8-bit integers (INT8), or even 4-bit integers (INT4).

## Why Quantize?

The primary motivations for quantization are to:

1.  **Reduce Model Size:** By using fewer bits to store each parameter, the total size of the model on disk and in memory can be significantly reduced. A model quantized from FP32 to INT8 will be roughly 4 times smaller.

2.  **Increase Inference Speed:** Many modern CPUs and specialized hardware (like GPUs and TPUs) can perform integer arithmetic much faster than floating-point arithmetic. This leads to lower inference latency, which is critical for real-time applications.

3.  **Improve Energy Efficiency:** Integer operations consume less power than floating-point operations. This is a major advantage for deploying models on edge devices with limited battery life.

## How Does It Work?

Quantization involves a mapping from the high-precision floating-point values to a smaller set of low-precision integer values. The simplest form is **linear quantization**, which uses a scaling factor and a zero-point to map the range of FP32 values to the range of the target integer type (e.g., `[0, 255]` for INT8).

-   **Weight Quantization:** This is the most common form of quantization, where only the model's weights are converted to a lower-precision format. The activations (the outputs of each layer) are still processed as floating-point numbers.

-   **Activation Quantization:** For further performance gains, the activations can also be quantized. This is known as "fully quantized" or "integer-only" inference.

-   **Quantization-Aware Training (QAT):** This technique simulates the effect of quantization during the training process. It allows the model to adapt to the lower precision, often resulting in better accuracy compared to post-training quantization.

-   **Post-Training Quantization (PTQ):** This is the simplest approach, where a pre-trained FP32 model is converted to a lower-precision format without any retraining. It's faster than QAT but may lead to a larger drop in accuracy.

This project focuses on **Post-Training Quantization (PTQ)** of large language models.
