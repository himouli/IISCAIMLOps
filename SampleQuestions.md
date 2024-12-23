### Model Evaluation and Bias-Variance Tradeoff

1. **What does it indicate if a model has low training error but high validation error?**
   - **Answer:** The model is overfitting.
   - **Explanation:** The model is memorizing the training data but failing to generalize to unseen data.

2. **Explain the relationship between high bias and underfitting.**
   - **Answer:** High bias leads to underfitting.
   - **Explanation:** The model is too simplistic to capture the underlying patterns in the data.

3. **What can be inferred if a model has low bias but high variance?**
   - **Answer:** The model is overfitting.
   - **Explanation:** It captures noise and variations in the training data, resulting in poor generalization.

4. **Why does k-fold cross-validation improve model generalization?**
   - **Answer:** It evaluates the model on multiple subsets of data.
   - **Explanation:** This reduces overfitting and provides a robust estimate of performance.

5. **Describe a scenario where a high training error occurs with a low validation error.**
   - **Answer:** Data leakage or mislabeled training data.
   - **Explanation:** Issues with the training set can lead to poor training performance.

### Loss Functions

6. **In which scenarios is binary cross-entropy preferred over categorical cross-entropy?**
   - **Answer:** Binary classification problems.
   - **Explanation:** BCE calculates loss for two-class problems, while CCE is for multi-class tasks.

7. **What type of loss function would you use for a multi-class classification problem with imbalanced data?**
   - **Answer:** Weighted categorical cross-entropy.
   - **Explanation:** It assigns weights to balance the importance of each class.

8. **Why is Mean Squared Error (MSE) commonly used for regression tasks?**
   - **Answer:** It penalizes larger errors more heavily.
   - **Explanation:** MSE captures the average squared differences between predicted and actual values.

9. **Describe how hinge loss is used in support vector machines.**
   - **Answer:** It penalizes predictions outside the margin.
   - **Explanation:** Hinge loss ensures the decision boundary is maximized.

10. **What is the role of a softmax activation function in cross-entropy loss?**
    - **Answer:** Converts logits to probabilities.
    - **Explanation:** Softmax ensures the output sums to 1, suitable for probability-based loss functions.

### Neural Network Architectures

11. **What is the significance of a 1x1 convolution in deep learning models?**
    - **Answer:** Dimensionality reduction and channel mixing.
    - **Explanation:** It changes the depth while retaining spatial information.

12. **How does max pooling reduce computational complexity in CNNs?**
    - **Answer:** By downsampling feature maps.
    - **Explanation:** Max pooling reduces spatial dimensions, decreasing computational load.

13. **What are the primary advantages of residual connections in deep networks?**
    - **Answer:** Prevent vanishing gradients and improve training.
    - **Explanation:** Residuals allow gradients to flow directly through shortcut paths.

14. **Why are transposed convolutions essential for autoencoders?**
    - **Answer:** For upsampling.
    - **Explanation:** Transposed convolutions increase spatial dimensions in the decoder.

15. **Explain the concept of depthwise separable convolutions and their impact on model efficiency.**
    - **Answer:** Split convolutions into depthwise and pointwise steps.
    - **Explanation:** They reduce computation and parameters compared to standard convolutions.

### Transfer Learning

16. **How does transfer learning improve performance in low-data scenarios?**
    - **Answer:** By leveraging pre-trained features.
    - **Explanation:** Transfer learning utilizes knowledge from larger datasets.

17. **Why is it important to fine-tune only the later layers of a pre-trained model in certain cases?**
    - **Answer:** To adapt to the specific task.
    - **Explanation:** Earlier layers capture generic features, while later layers specialize in task-specific features.

18. **Describe the difference between feature extraction and fine-tuning in transfer learning.**
    - **Answer:** Feature extraction uses frozen layers; fine-tuning updates weights.
    - **Explanation:** Feature extraction preserves pre-trained knowledge, while fine-tuning adapts it.

19. **In what situations would transfer learning be less effective?**
    - **Answer:** When the target task is vastly different from the source task.
    - **Explanation:** Pre-trained features may not generalize well to unrelated domains.

20. **Name three popular pre-trained models for transfer learning in image classification.**
    - **Answer:** VGG, ResNet, Inception.
    - **Explanation:** These models are trained on large datasets like ImageNet.

### Regularization and Optimization

21. **What is the purpose of L1 and L2 regularization in machine learning models?**
    - **Answer:** To prevent overfitting.
    - **Explanation:** L1 induces sparsity, while L2 penalizes large weights.

22. **Why does dropout help prevent overfitting in neural networks?**
    - **Answer:** By randomly dropping neurons during training.
    - **Explanation:** This forces the network to learn robust features.

23. **Explain how batch normalization accelerates model training.**
    - **Answer:** By normalizing layer inputs.
    - **Explanation:** It stabilizes learning and reduces dependence on initialization.

24. **What is the significance of the Adam optimizer in training deep networks?**
    - **Answer:** Adaptive learning rates.
    - **Explanation:** Adam combines momentum and adaptive learning to converge faster.

25. **How do gradient clipping techniques address exploding gradients?**
    - **Answer:** By capping gradients at a threshold.
    - **Explanation:** This prevents gradients from growing too large during backpropagation.

### Autoencoders and Feature Extraction

26. **What are the key components of an autoencoder, and how do they work together?**
    - **Answer:** Encoder, decoder, and bottleneck.
    - **Explanation:** The encoder compresses data, the bottleneck represents it, and the decoder reconstructs it.

27. **Describe the difference between a variational autoencoder and a regular autoencoder.**
    - **Answer:** Variational autoencoders generate probabilistic outputs.
    - **Explanation:** They learn distributions instead of deterministic mappings.

28. **What role does dimensionality reduction play in feature extraction?**
    - **Answer:** Reduces noise and irrelevant features.
    - **Explanation:** It focuses on meaningful patterns in the data.

29. **Why is the decoder component critical in image reconstruction tasks?**
    - **Answer:** Reconstructs the original input.
    - **Explanation:** It maps compressed representations back to high-dimensional data.

30. **How can autoencoders be used for anomaly detection?**
    - **Answer:** By identifying high reconstruction errors.
    - **Explanation:** Anomalies often deviate from learned patterns.

### Image Classification and Object Detection

31. **How does Intersection Over Union (IoU) measure performance in object detection?**
    - **Answer:** It quantifies overlap between predicted and true bounding boxes.
    - **Explanation:** Higher IoU indicates better detection accuracy.

32. **What are the key differences between object detection and semantic segmentation?**
    - **Answer:** Detection localizes objects; segmentation classifies each pixel.
    - **Explanation:** Object detection focuses on bounding boxes, while segmentation provides fine-grained labels.

33. **Describe the purpose of anchor boxes in object detection algorithms.**
    - **Answer:** Predefined boxes for bounding box prediction.
    - **Explanation:** Anchor boxes simplify the localization of objects at different scales and aspect ratios.

34. **Why is non-max suppression necessary in object detection pipelines?**
    - **Answer:** To remove redundant bounding boxes.
    - **Explanation:** Non-max suppression keeps the most confident predictions.

35. **How does a fully convolutional network differ from a regular CNN in segmentation tasks?**
    - **Answer:** It produces pixel-wise outputs.
    - **Explanation:** FCNs replace dense layers with convolutional layers for spatial output.

### Confusion Matrix and Metrics

36. **Explain the difference between precision and recall.**
    - **Answer:** Precision measures correct positives; recall measures retrieved positives.
    - **Explanation:** Precision focuses on accuracy, while recall focuses on completeness.

37. **How is the F1-score useful in evaluating model performance?**
    - **Answer:** It balances precision and recall.
    - **Explanation:** F1-score is the harmonic mean of precision and recall.

38. **What does a high false-positive rate indicate about a classifier?**
    - **Answer:** The classifier is overly sensitive.
    - **Explanation:** It predicts positive too often, even when incorrect.

39. **Describe a situation where the ROC-AUC metric would be more informative than accuracy.**
    - **Answer:** When data is imbalanced.
    - **Explanation:** ROC-AUC evaluates performance across all classification thresholds.

40. **How do you compute precision for a specific class in a multi-class classification problem?**
    - **Answer:** Divide true positives by predicted positives for that class.
    - **Explanation:** Precision is calculated independently for each class.

### Miscellaneous

41. **What is the purpose of a Flatten layer in neural network architectures?**
    - **Answer:** Converts feature maps to vectors.
    - **Explanation:** Flatten prepares data for dense layers.

42. **Explain how stride and padding affect the output dimensions in convolutional layers.**
    - **Answer:** Stride reduces dimensions; padding preserves them.
    - **Explanation:** Larger strides decrease size, while padding prevents shrinkage.

43. **Why is data augmentation crucial for training image classification models?**
    - **Answer:** Increases data variability.
    - **Explanation:** It improves generalization by creating diverse samples.

44. **What is the significance of activations in visualizing CNN filters?**
    - **Answer:** Show what the filter detects.
    - **Explanation:** Activations reveal patterns learned by filters.

45. **Describe how label smoothing helps in training deep learning models.**
    - **Answer:** Reduces overconfidence.
    - **Explanation:** It softens labels to improve generalization.

46. **How do GANs generate new data samples similar to the training set?**
    - **Answer:** By learning a data distribution.
    - **Explanation:** GANs use a generator and discriminator in adversarial training.

47. **What is the role of the discriminator in a GAN architecture?**
    - **Answer:** Distinguishes real from fake data.
    - **Explanation:** It trains the generator to improve data realism.

48. **Why is the choice of kernel size critical in designing convolutional networks?**
    - **Answer:** Affects feature detection.
    - **Explanation:** Larger kernels capture broader patterns, while smaller kernels detect fine details.

49. **How do learning rate schedulers improve model convergence?**
    - **Answer:** Adjust the learning rate dynamically.
    - **Explanation:** They balance fast convergence and stable training.

50. **Explain the significance of the output layer's activation function in a neural network.**
    - **Answer:** Determines output type.
    - **Explanation:** For example, softmax produces probabilities, while sigmoid outputs binary values.

