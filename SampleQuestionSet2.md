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

6. **What causes overfitting in a neural network?**
   - **Answer:** Excessive model complexity or lack of sufficient training data.
   - **Explanation:** The model learns noise and details in the training data that do not generalize to unseen data.

7. **How can early stopping help prevent overfitting?**
   - **Answer:** By halting training when validation performance stops improving.
   - **Explanation:** It ensures the model does not overtrain on the training dataset.

8. **What is the primary difference between train-test split and k-fold cross-validation?**
   - **Answer:** Train-test split uses a single division, while k-fold cross-validation averages results over multiple splits.
   - **Explanation:** K-fold cross-validation provides a more robust performance evaluation.

9. **What does it mean if a model has low training error and low validation error?**
   - **Answer:** The model is well-fit.
   - **Explanation:** It indicates that the model captures the underlying patterns and generalizes well.

10. **How do you identify high variance in a machine learning model?**
    - **Answer:** Large differences between training and validation errors.
    - **Explanation:** High variance models perform well on training data but poorly on validation data.

### Loss Functions

11. **Why is cross-entropy loss preferred for classification problems?**
    - **Answer:** It measures the distance between predicted probabilities and actual labels.
    - **Explanation:** Cross-entropy penalizes incorrect predictions more heavily.

12. **What loss function would you use for a regression task predicting house prices?**
    - **Answer:** Mean Squared Error (MSE).
    - **Explanation:** MSE captures the squared differences between predicted and actual values.

13. **How does Huber loss differ from MSE?**
    - **Answer:** Huber loss is less sensitive to outliers.
    - **Explanation:** It combines MSE for small errors and MAE for large errors.

14. **What is the primary purpose of label smoothing in loss functions?**
    - **Answer:** To reduce overconfidence in predictions.
    - **Explanation:** It assigns a small probability to incorrect classes to improve generalization.

15. **Why is hinge loss suitable for SVMs?**
    - **Answer:** It maximizes the margin between classes.
    - **Explanation:** Hinge loss ensures predictions are confidently on the correct side of the decision boundary.

### Neural Network Architectures

16. **What is the purpose of activation functions in neural networks?**
    - **Answer:** To introduce non-linearity.
    - **Explanation:** Activation functions allow the network to model complex relationships.

17. **Why is ReLU widely used as an activation function?**
    - **Answer:** It avoids vanishing gradients.
    - **Explanation:** ReLU outputs zero for negative inputs and the input itself for positive values, ensuring efficient gradient flow.

18. **How does a CNN handle spatial information in images?**
    - **Answer:** By using convolutional layers.
    - **Explanation:** Convolutions detect features like edges and textures in local regions of the image.

19. **What is the effect of using larger strides in a convolutional layer?**
    - **Answer:** Reduces the spatial dimensions of the output.
    - **Explanation:** Larger strides skip more pixels during the convolution operation.

20. **What is the significance of global average pooling in CNNs?**
    - **Answer:** Reduces each feature map to a single value.
    - **Explanation:** It replaces fully connected layers to reduce parameters and improve generalization.

### Transfer Learning

21. **How does freezing layers in transfer learning affect training?**
    - **Answer:** Prevents the weights of pre-trained layers from being updated.
    - **Explanation:** It allows the model to retain general features learned from the source task.

22. **Why is ImageNet commonly used for transfer learning?**
    - **Answer:** It provides a large and diverse dataset.
    - **Explanation:** Pre-trained models on ImageNet capture general features applicable to many tasks.

23. **How can transfer learning improve computational efficiency?**
    - **Answer:** By reusing pre-trained weights.
    - **Explanation:** It reduces the need for training from scratch, saving time and resources.

24. **What is the key advantage of domain adaptation in transfer learning?**
    - **Answer:** Adapts a model trained on a source domain to a target domain.
    - **Explanation:** Domain adaptation reduces the gap between source and target distributions.

25. **What is the role of fine-tuning in transfer learning?**
    - **Answer:** To adapt pre-trained models to a specific task.
    - **Explanation:** Fine-tuning updates weights in the later layers to specialize in the target task.

### Regularization and Optimization

26. **What is the role of weight decay in regularization?**
    - **Answer:** Penalizes large weights.
    - **Explanation:** Weight decay discourages overfitting by adding a penalty proportional to the weights’ magnitude.

27. **How does dropout prevent overfitting?**
    - **Answer:** By randomly dropping neurons during training.
    - **Explanation:** This forces the network to learn robust features from diverse subsets of neurons.

28. **Why is batch normalization used in deep networks?**
    - **Answer:** To normalize activations.
    - **Explanation:** Batch normalization stabilizes training and accelerates convergence.

29. **What is the primary function of learning rate schedules?**
    - **Answer:** To adjust the learning rate during training.
    - **Explanation:** Learning rate schedules balance fast convergence and stable optimization.

30. **Why is the Adam optimizer popular for training deep learning models?**
    - **Answer:** It combines momentum and adaptive learning rates.
    - **Explanation:** Adam adapts to individual parameter updates for efficient optimization.

### Autoencoders and Feature Extraction

31. **What is the purpose of the encoder in an autoencoder?**
    - **Answer:** To compress input data.
    - **Explanation:** The encoder maps high-dimensional input data to a low-dimensional representation.

32. **What is a variational autoencoder designed for?**
    - **Answer:** Generating new data samples.
    - **Explanation:** Variational autoencoders learn a latent distribution to sample and reconstruct data.

33. **Why are autoencoders suitable for denoising tasks?**
    - **Answer:** They learn to reconstruct clean data from noisy input.
    - **Explanation:** The network captures the underlying structure of the data, removing noise during reconstruction.

34. **How is feature extraction performed using pre-trained models?**
    - **Answer:** By using the output of intermediate layers.
    - **Explanation:** Intermediate layers capture meaningful features that can be used in downstream tasks.

35. **What is the bottleneck in an autoencoder?**
    - **Answer:** A compressed representation of the input data.
    - **Explanation:** The bottleneck forces the model to learn efficient, high-level features.

### Image Classification and Object Detection

36. **What is the purpose of anchor boxes in object detection?**
    - **Answer:** To detect objects at different scales and aspect ratios.
    - **Explanation:** Anchor boxes provide predefined bounding boxes to localize objects efficiently.

37. **How does non-max suppression improve object detection?**
    - **Answer:** By removing redundant bounding boxes.
    - **Explanation:** It retains only the most confident prediction for each object.

38. **What is the role of Intersection Over Union (IoU) in object detection?**
    - **Answer:** To measure overlap between predicted and ground truth boxes.
    - **Explanation:** IoU evaluates the accuracy of object localization.

39. **How do fully convolutional networks (FCNs) perform image segmentation?**
    - **Answer:** By producing pixel-wise predictions.
    - **Explanation:** FCNs replace dense layers with convolutional layers to preserve spatial information.

40. **What is the main challenge in semantic segmentation?**
    - **Answer:** Classifying each pixel accurately.
    - **Explanation:** It requires both spatial localization and fine-grained classification.

### Confusion Matrix and Metrics

41. **How is precision calculated in a binary classification problem?**
    - **Answer:** True Positives / (True Positives + False Positives).
    - **Explanation:** Precision measures the accuracy of positive predictions.

42. **What does a high recall value indicate?**
    - **Answer:** Most positive instances are correctly identified.
    - **Explanation:** Recall focuses on completeness in positive class identification.

43. **How is the F1-score calculated?**
    - **Answer:** 2 × (Precision × Recall) / (Precision + Recall).
    - **Explanation:** The F1-score balances precision and recall.

44. **What is the purpose of the ROC-AUC metric?**
    - **Answer:** To evaluate classification performance across thresholds.
    - **Explanation:** ROC-AUC measures the trade-off between true positives and false positives.

45. **Why is accuracy not reliable for imbalanced datasets?**
    - **Answer:** It can be biased towards the majority class.
    - **Explanation:** High accuracy may occur despite poor performance on the minority class.

### Miscellaneous

46. **What is the purpose of data augmentation in training image models?**
    - **Answer:** To increase the diversity of training data.
    - **Explanation:** Data augmentation generates variations of existing samples, improving generalization.

47. **Why are one-hot encodings used in classification tasks?**
    - **Answer:** To represent categorical labels.
    - **Explanation:** One-hot encoding creates a binary vector for each class label.

48. **How does the softmax function differ from sigmoid in multi-class classification?**
    - **Answer:** Softmax normalizes probabilities across all classes.
    - **Explanation:** Sigmoid is used for binary classification, while softmax outputs probabilities for multiple classes.

49. **What is the role of dropout in neural networks?**
    - **Answer:** To prevent overfitting.
    - **Explanation:** Dropout randomly deactivates neurons during training, forcing the network to learn redundant features.

50. **How does a Flatten layer work in a CNN?**
    - **Answer:** Converts feature maps into vectors.
    - **Explanation:** Flatten prepares data for fully connected layers in classification tasks.

