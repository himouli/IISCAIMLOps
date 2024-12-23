**Sample Quiz: Machine Learning and Computer Vision**

**Instructions:**
- Answer all the questions.
- Each question carries equal marks.
- Select the most appropriate answer for multiple-choice questions.

---

### **Quiz Questions**

#### **Part A: Multiple-Choice Questions**

1. A regression model is trained on a dataset. During evaluation, it exhibits high training error and high validation error. Which of the following best describes the model's behavior?
   - A) Overfitting with low bias and high variance
   - B) Underfitting with high bias and low variance
   - C) Balanced bias-variance tradeoff
   - D) Overfitting with high bias and low variance

2. Which loss function is most appropriate for a binary classification task?
   - A) Mean Squared Error
   - B) Binary Cross-Entropy
   - C) Categorical Cross-Entropy
   - D) Hinge Loss

3. You are designing a ConvNet with the following parameters: Input size = \(32 \times 32 \times 3\), Filter size = \(3 \times 3\), Stride = 1, Padding = 'valid', Filters = 16. What is the output size after the convolution?
   - A) \(32 \times 32 \times 16\)
   - B) \(30 \times 30 \times 16\)
   - C) \(16 \times 16 \times 16\)
   - D) \(15 \times 15 \times 16\)

4. What is the primary advantage of using dropout in a neural network?
   - A) It increases training speed
   - B) It reduces the number of parameters
   - C) It prevents overfitting
   - D) It helps normalize activations

5. A pre-trained ResNet model is used for transfer learning on a new dataset. Which layers are most commonly fine-tuned?
   - A) Convolutional layers
   - B) Fully connected layers
   - C) Early residual blocks
   - D) Pooling layers

6. Which of the following metrics is most suitable for evaluating object detection performance?
   - A) Mean Absolute Error (MAE)
   - B) Intersection Over Union (IoU)
   - C) ROC-AUC
   - D) Precision-Recall Curve

7. What does a high F1-score indicate in a classification model?
   - A) High precision and low recall
   - B) High recall and low precision
   - C) A balance between precision and recall
   - D) High accuracy only

8. Which technique is commonly used to increase the diversity of training data for image classification?
   - A) Batch Normalization
   - B) Data Augmentation
   - C) Gradient Clipping
   - D) Regularization

9. In object detection, what is the role of anchor boxes?
   - A) To normalize bounding boxes
   - B) To handle objects of varying scales and aspect ratios
   - C) To reduce false positives
   - D) To calculate IoU

10. A \(64 \times 64 \times 3\) image passes through the following layers:
    - Conv2D: \(3 \times 3\), Stride = 1, Filters = 32, Padding = 'same'
    - MaxPooling: \(2 \times 2\), Stride = 2
    What is the output size after the MaxPooling layer?
    - A) \(32 \times 32 \times 32\)
    - B) \(64 \times 64 \times 32\)
    - C) \(31 \times 31 \times 32\)
    - D) \(16 \times 16 \times 32\)

#### **Part B: Short Answers**

11. Explain why ReLU is widely used as an activation function in neural networks.

12. Define the purpose of the bottleneck layer in an autoencoder.

13. Why is it important to freeze certain layers in a pre-trained model during transfer learning?

14. Describe the primary difference between semantic segmentation and instance segmentation.

15. What is the significance of the IoU metric in object detection?

---

### **Answer Key**

#### **Part A: Multiple-Choice Questions**

1. **B** - Underfitting with high bias and low variance.
   - Explanation: High training and validation errors indicate the model is too simplistic to capture patterns.

2. **B** - Binary Cross-Entropy.
   - Explanation: It measures the logarithmic loss for binary classification problems.

3. **B** - \(30 \times 30 \times 16\).
   - Explanation: Output size = \((W - K)/S + 1 = (32 - 3)/1 + 1 = 30\).

4. **C** - It prevents overfitting.
   - Explanation: Dropout randomly deactivates neurons during training, forcing robust feature learning.

5. **B** - Fully connected layers.
   - Explanation: These layers are specific to the pre-trained model's original task and need adaptation.

6. **B** - Intersection Over Union (IoU).
   - Explanation: IoU evaluates overlap between predicted and ground truth bounding boxes.

7. **C** - A balance between precision and recall.
   - Explanation: F1-score is the harmonic mean of precision and recall.

8. **B** - Data Augmentation.
   - Explanation: Techniques like flipping, cropping, and rotation artificially expand the dataset.

9. **B** - To handle objects of varying scales and aspect ratios.
   - Explanation: Anchor boxes enable the detection of objects at different sizes and shapes.

10. **A** - \(32 \times 32 \times 32\).
    - Explanation: MaxPooling halves the spatial dimensions while retaining the depth.

#### **Part B: Short Answers**

11. **ReLU Activation:**
    - ReLU introduces non-linearity, allowing networks to learn complex patterns. It avoids the vanishing gradient problem by outputting zero for negative inputs and the input itself for positive values.

12. **Bottleneck in Autoencoder:**
    - The bottleneck compresses input data into a lower-dimensional latent representation, capturing essential features for reconstruction.

13. **Freezing Layers in Transfer Learning:**
    - Early layers capture general features (e.g., edges, textures) and do not require retraining. Freezing them reduces training time and prevents overfitting.

14. **Semantic vs. Instance Segmentation:**
    - Semantic segmentation labels every pixel with a class but does not distinguish between instances. Instance segmentation identifies individual object instances within the same class.

15. **Significance of IoU:**
    - IoU measures the overlap between predicted and ground truth bounding boxes, ensuring accurate object localization and detection.

---

This simulated question paper and answer booklet provide a complete practice set for quiz preparation. Let me know if you need more!

