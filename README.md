# ESP32 Smart Pendant — Fall Detection (Edge ML)

An end-to-end fall detection system for an ESP32-S3 smart pendant using motion sensors and an on-device TensorFlow Lite model, optimized for real-time inference with low latency and low power consumption. [1]

## Contents
- Overview and goals. [1]
- Hardware and sensors. [1]
- Dataset and labeling. [1]
- Preprocessing pipeline. [1]
- Model training and class balancing. [1]
- Evaluation and metrics (TestA/TestB).
- Threshold tuning on validation.
- TFLite export (float and int8) and validation parity.
- Deployment configuration and on-device smoothing. [1]
- Repository structure and how to reproduce. [1]

## 1) Overview
This project implements reliable fall detection on an ESP32-S3 pendant using a compact neural network that analyzes short windows of motion data at 50 Hz and raises an alert when a fall event is detected. [1]
The pipeline covers data collection, preprocessing, model training and tuning, conversion to TensorFlow Lite, and deployment with postprocessing to minimize false alarms during daily activities. [1]

## 2) Hardware
- Microcontroller: ESP32-S3, selected for its vector instructions and embedded ML suitability. [1]
- Motion sensor: 6-axis accelerometer + gyroscope (e.g., MPU6050/MPU6886), sampled at approximately 50 Hz for balanced responsiveness and compute cost. [1]
- Feedback: Vibration motor or buzzer to notify the wearer and allow cancel before escalating an alert, powered by a battery for wearable use. [1]

## 3) Project goals
- Detect falls accurately while keeping false alarms low in daily activities. [1]
- Run inference on-device within strict compute and memory constraints suitable for continuous wearable monitoring. [1]
- Achieve responsive detection with inference and decision latency well under a second for comfort and safety. [1]

## 4) Dataset
- Source: Labeled motion windows captured from the wearable or equivalent sensor setup, stored as arrays for training and validation, and organized into Train, Val, and two held-out tests (TestA, TestB) to measure generalization. [1]
- Labels: Binary classification with 0 = non-fall (activities of daily living) and 1 = fall, enabling clear evaluation with precision, recall, F1, and confusion matrices. [1]
- Storage format: Preprocessed NumPy arrays saved as X_train/X_val/X_testA/X_testB for inputs and y_* for labels, aligned with a preprocessing_manifest.json describing sampling rate, window size, and normalization statistics. [1]

## 5) Labeling protocol
- Falls are annotated when a rapid change in motion magnitude and characteristic post-impact patterns are observed, while non-falls include routine movements like walking, sitting, and transitions. [1]
- Labels are validated by cross-checking event timestamps and signal signatures across axes and magnitude to reduce ambiguity during model training. [1]
- The binary setup simplifies deployment decisions while enabling conservative postprocessing on-device to mitigate spurious triggers. [1]

## 6) Preprocessing
- Resampling: Motion data is resampled or recorded at fs_out ≈ 50 Hz to unify timing and reduce high-frequency noise, reflected in the preprocessing manifest. [1]
- Windowing: Fixed-length windows (e.g., 1 second at 50 samples) are extracted with a chosen hop step to balance temporal resolution and training sample count. [1]
- Channels: Feature channels include per-axis acceleration and derived magnitude, with channel order recorded to ensure training and deployment match. [1]
- Standardization: Per-channel mean and standard deviation from the manifest are used to normalize windows, and the same values must be applied on-device before inference. [1]

## 7) Repository structure
- data/prepared_dataset_50hz/: Preprocessed NumPy arrays and preprocessing_manifest.json used by training and deployment. [1]
- notebooks/train.ipynb: End-to-end training, integrity checks, dataset summaries, threshold tuning, and export to TFLite. [1]
- models/: Saved Keras model best_cnn_fall.keras and exported TFLite models best_cnn_fall.tflite (float) and best_cnn_fall_int8.tflite (int8). [1]
- deploy_config.json: Deployment-time configuration merged from the manifest with model threshold and smoothing parameters for the device. [1]

## 8) Training setup
- Input shape: Keras Conv1D expects data shaped (N, T, C) with N windows, T time steps, and C channels as recorded in the manifest. [1]
- Model family: Lightweight 1D CNN designed for embedded inference and optimized for short windows, with careful parameter count to meet memory constraints. [1]
- Reproducibility: Fixed random seeds are used in NumPy and TensorFlow to stabilize results across runs for debugging and ablation studies. [1]

## 9) Class balance and weights
- Class counts are computed from y_train to quantify the imbalance between falls and non-falls. [1]
- A balanced heuristic weight is applied during training: class_weight[c] = total / (2 × count_c), which up-weights the rarer class to improve recall without heavily penalizing precision. [1]
- Balanced training helps prevent trivial majority-class predictions and improves minority detection at reasonable thresholds. [1]

## 10) Data pipeline
- Datasets are created via tf.data.Dataset.from_tensor_slices and optionally shuffled for training with a fixed seed, then batched and prefetched for throughput. [1]
- Validation and test splits are not shuffled to preserve deterministic evaluation and traceability of any misclassifications. [1]
- Fixed BATCH_SIZE and AUTOTUNE are used to balance memory usage and speed, especially when iterating threshold tuning. [1]

## 11) Evaluation (default threshold 0.50)
- TestA (0.50): Accuracy ≈ 0.713 with confusion matrix [, ], precision-recall AUC 0.769, indicating a good balance but with some false alarms and some missed falls.   
- TestB (0.50): Accuracy ≈ 0.625 with confusion matrix [, ], precision-recall AUC 0.431, showing high false positives in more varied conditions and the need for tuning.   
- Interpretation: The default threshold catches many falls but triggers too often on non-fall windows in broader scenarios, motivating validation-based threshold tuning. 

## 12) Threshold tuning (validation-driven)
- Validation search computes precision-recall curves and picks the threshold that maximizes F1 for a balanced trade-off, returning threshold, precision, recall, and F1 at the selected operating point.   
- Best threshold on validation is approximately 0.649, with precision ≈ 0.811, recall ≈ 0.763, and F1 ≈ 0.787, indicating strong balance without excessive false positives.   
- This tuned threshold improves TestA accuracy to ≈ 0.742 and TestB accuracy to ≈ 0.752 by reducing false positives while maintaining high recall. 

## 13) Evaluation (tuned threshold ~0.65)
- TestA (0.65): Confusion matrix [, ] with PR AUC unchanged at 0.769 since AUC is threshold-independent, trading a small recall drop for fewer false alarms.   
- TestB (0.65): Confusion matrix [, ], showing a substantial reduction in false positives and a practical improvement for deployment scenarios.   
- Takeaway: Validation-driven thresholding yields a better operating point for the wearable, balancing safety (recall) and user comfort (precision). 

## 14) Model export and quantization
- The best Keras model is saved as best_cnn_fall.keras and converted to TensorFlow Lite: a float32 model best_cnn_fall.tflite and an int8-quantized model best_cnn_fall_int8.tflite using Optimize.DEFAULT.   
- Float vs int8: The int8 model is typically 3–4× smaller and faster with minor, often negligible, accuracy differences, making it preferable for ESP32 deployment with constrained memory and power.   
- File sizes are printed in kilobytes to verify the storage footprint before flashing to the device or bundling in firmware. 

## 15) TFLite validation parity
- Validation at threshold 0.65 shows Keras and TFLite-float produce identical metrics on the validation set, confirming correct conversion.   
- TFLite-int8 matches within a 0.1% accuracy delta, indicating negligible loss from dynamic-range quantization for this task and enabling efficient on-device inference.   
- A deprecation warning suggests migrating from tf.lite.Interpreter to the LiteRT interpreter in future TF versions, without affecting current results. 

## 16) Deployment configuration
- A single deploy_config.json merges model and preprocessing essentials for the firmware: model filename, input shape, threshold, sampling rate, window parameters, channel order, and normalization statistics. [1]
- Postprocessing includes confirm_windows (e.g., 2 consecutive positive windows required) and refractory_windows (e.g., wait 2 windows after a trigger) to stabilize alerts. [1]
- The device reads this configuration at startup to ensure on-device preprocessing exactly matches training-time transforms for consistent inference. [1]

## 17) On-device flow
- Acquire motion samples continuously at fs_out, maintain a sliding window of window_size with hop step, and standardize using channel_mean and channel_std from the config. [1]
- Run inference on each window with the int8 TFLite model and compare the probability against the tuned threshold to produce a binary decision per window. [1]
- Apply postprocessing: require confirm_windows positives to trigger the vibration alert and enforce refractory_windows to prevent duplicate alerts during recovery. [1]

## 18) How to reproduce
- Prepare  Generate prepared_dataset_50hz/ with X_*.npy arrays and preprocessing_manifest.json describing fs_out, channels, window_size, step, mean, and std. [1]
- Train: Run the training notebook to load splits, compute class weights, build the model, train with validation monitoring, and save best_cnn_fall.keras. [1]
- Evaluate and tune: Evaluate on TestA/TestB at 0.50, tune the threshold on Val for best F1, and re-evaluate tests at the tuned threshold to confirm improvements.   
- Export: Convert to best_cnn_fall.tflite and best_cnn_fall_int8.tflite and verify parity on validation using the interpreter-based script.   
- Configure deployment: Write deploy_config.json with model threshold and smoothing parameters and flash the firmware to the ESP32-S3. [1]

## 19) Known limitations and next steps
- TestB at the default threshold showed high false positives, mitigated by tuning, suggesting further data augmentation and environment diversity will improve generalization.   
- Consider extending inputs to include gyroscope-derived features or adding temporal models if latency and memory budgets allow. [1]
- Explore quantization-aware training or representative dataset calibration if moving from dynamic-range quantization to full int8 inference becomes necessary. [1]

## 20) Files and artifacts
- best_cnn_fall.keras: Best Keras checkpoint restored by early stopping for export.   
- best_cnn_fall.tflite: Float32 TFLite model for validation parity testing.   
- best_cnn_fall_int8.tflite: Int8 dynamic-range quantized model for on-device deployment.   
- deploy_config.json: Consolidated model, preprocessing, and postprocessing settings consumed by the firmware. [1]
- prepared_dataset_50hz/: Numpy arrays for splits and preprocessing_manifest.json with fs_out, channels, window_size, step, mean, std. [1]

## 21) Safety and UX behavior
- When a fall is detected, the device vibrates and awaits user cancellation; if not canceled, the system escalates the alert for downstream handling. [1]
- Confirm and refractory windows avoid alert flapping and reduce repeated triggers during post-impact immobility or recovery. [1]
- Threshold tuning prioritizes a balance that reduces nuisance alerts while keeping high sensitivity to true falls for safety. 

## 22) License and acknowledgements
- This project references a generic edge ML pipeline for wearable fall detection; adapt licensing and attributions to your organization’s standards. [1]
- Acknowledge contributors for data collection, labeling, modeling, firmware development, and field testing as applicable. [1]

---

### Appendix: Key numbers at a glance
- Validation-tuned threshold: approximately 0.649 with precision ≈ 0.811, recall ≈ 0.763, F1 ≈ 0.787.   
- TestA vs TestB performance before and after tuning shows improved accuracy and reduced false positives at the tuned threshold for practical deployment.   
- TFLite float vs int8 on validation: near-identical metrics, confirming safe adoption of int8 for ESP32-S3 deployment.

### Sources
[1] Task-1_-Fall-Detection-Feature-Edge-ML-ESP32-Smart-Pendant.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_e363f051-249d-4301-9765-fd18cdedee00/7717b2f6-6490-4db4-ba42-6b31b5680b36/Task-1_-Fall-Detection-Feature-Edge-ML-ESP32-Smart-Pendant.pdf
