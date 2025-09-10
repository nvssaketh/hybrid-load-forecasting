# A Comparative Study of Hybrid RNN-SNN and DRNN-SNN Models for Electric Load Forecasting

## Overview
This repository chronicles a project on high-accuracy electricity load forecasting by developing and comparing two distinct hybrid neural network architectures: a baseline RNN+SNN model and an enhanced DRNN+SNN model. The initial RNN+SNN hybrid combines a standard Recurrent Neural Network, effective for general sequence modeling, with a bio-inspired Spiking Neural Network (SNN) that processes temporal data through event-driven spikes. The subsequent, more advanced DRNN+SNN model substitutes the standard RNN with a Dilated RNN (DRNN), specifically chosen for its superior ability to capture long-range dependencies and multi-scale patterns within the time-series data. By methodically evaluating both hybrid systems, this project demonstrates a clear performance improvement, showcasing how architectural enhancements from RNN to DRNN within a hybrid framework can lead to a more robust and accurate forecasting solution.

The RNN-SNN hybrid, combining recurrent neural networks with spiking neural networks, achieved a MAPE of 2.92%. The DRNN-SNN hybrid, which incorporates dilated recurrence for multi-scale temporal dependencies, achieved a superior MAPE of 2.11%, demonstrating a significant improvement in forecast accuracy.

## Key Features
- Implementation of bio-inspired Spiking Neural Networks (SNNs) for regression tasks.
- Development of adaptive hybrid weighting strategies.
- Comparative evaluation with industry-standard metrics (MAPE, MAE, R²).
- Open-source codebase for reproducibility and future research.

## Performance Summary

| Model            | MAPE (%) | MAE     | R²   |
|-----------------|---------|--------|------|
| RNN-SNN Hybrid   | 2.92    | 1156.8 | 0.943 |
| DRNN-SNN Hybrid  | 2.11    | 876.5  | 0.971 |

![Hybrid Model Comparison](Hybrid_Model_Comparison.png)

## Methodology
- Data preprocessing involved outlier removal, normalization, and sequence generation.
- RNN architecture based on stacked LSTM layers.
- DRNN architecture introduced dilated LSTM cells to capture long-term temporal patterns.
- Spiking Neural Network (SNN) designed with Leaky Integrate-and-Fire neurons to enhance pattern recognition.
- Models were trained using adaptive optimization techniques like early stopping and gradient clipping.

## Results
- DRNN-SNN hybrid outperformed the standard RNN and RNN-SNN hybrid by achieving the best accuracy.
- The models demonstrated that combining temporal trends with bio-inspired mechanisms leads to improved forecasting.
- The final MAPE of 2.11% represents state-of-the-art performance suitable for real-world grid forecasting applications.

## Tools & Frameworks
- Python 3.x, PyTorch, TensorFlow, snnTorch
- scikit-learn, pandas, numpy, matplotlib
- Google Colab for GPU-accelerated training

## Repository
Access the complete source code, datasets, and documentation at:  
[https://github.com/nvssaketh/hybrid-load-forecasting](https://github.com/nvssaketh/hybrid-load-forecasting)

-
