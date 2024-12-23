**Practice Quiz: Machine Learning and Computer Vision**

**Instructions:**
- Attempt all questions.
- Select the most appropriate answer for multiple-choice questions.
- Write brief explanations for short-answer questions.

---

### **Quiz Questions**

#### **Part A: Multiple-Choice Questions**

1. You have a model that performs well on training data but poorly on validation data. What does this indicate?  
   - A) Underfitting  
   - B) Overfitting  
   - C) Balanced bias-variance tradeoff  
   - D) Data leakage  

2. Which of the following activation functions is most likely to suffer from the vanishing gradient problem?  
   - A) ReLU  
   - B) Sigmoid  
   - C) Softmax  
   - D) Leaky ReLU  

3. In a ConvNet, what is the effect of using padding set to 'same'?  
   - A) Reduces the spatial dimensions of the output  
   - B) Preserves the spatial dimensions of the input  
   - C) Increases the depth of the feature maps  
   - D) Adds dropout to the layer  

4. A regression task uses Mean Squared Error (MSE) as the loss function. What does a high MSE value indicate?  
   - A) Accurate predictions  
   - B) High variance in predictions  
   - C) Large differences between predicted and actual values  
   - D) A balanced model  

5. What is the primary purpose of anchor boxes in object detection?  
   - A) To segment objects in an image  
   - B) To handle objects of different sizes and shapes  
   - C) To reduce the need for data augmentation  
   - D) To calculate IoU scores  

6. In transfer learning, which of the following is commonly true?  
   - A) Early layers are frozen, and later layers are fine-tuned  
   - B) All layers are re-trained from scratch  
   - C) Pre-trained models are used only for text data  
   - D) Fully connected layers are frozen, and convolutional layers are trained  

7. Which metric is most suitable for evaluating an imbalanced binary classification problem?  
   - A) Accuracy  
   - B) F1-Score  
   - C) Mean Squared Error  
   - D) IoU  

8. What is the primary function of a pooling layer in a ConvNet?  
   - A) To increase spatial dimensions  
   - B) To reduce computational complexity and extract dominant features  
   - C) To increase the depth of feature maps  
   - D) To normalize activations  

9. Which type of regularization is used to encourage sparsity in model weights?  
   - A) L1 Regularization  
   - B) L2 Regularization  
   - C) Dropout  
   - D) Early stopping  

10. A \(128 \times 128 \times 3\) image passes through a \(3 \times 3\) Conv2D layer with 64 filters, stride = 2, and padding = 'valid'. What is the output size?  
    - A) \(64 \times 64 \times 64\)  
    - B) \(63 \times 63 \times 64\)  
    - C) \(126 \times 126 \times 64\)  
    - D) \(32 \times 32 \times 64\)  

---

#### **Part B: Short-Answer Questions**

11. Explain the purpose of the bottleneck layer in an autoencoder and its role in dimensionality reduction.  

12. What is Intersection Over Union (IoU), and why is it important in object detection tasks?  

13. Describe the difference between feature extraction and fine-tuning in transfer learning.  

14. Why is ReLU preferred over Sigmoid as an activation function in deep networks?  

15. What is the significance of non-max suppression in object detection models?  

---

### **Answer Key**

#### **Part A: Multiple-Choice Questions**

1. **B** - Overfitting.  
   - Explanation: The model memorizes training data but fails to generalize to unseen data.  

2. **B** - Sigmoid.  
   - Explanation: Sigmoid activation compresses gradients to a small range, leading to vanishing gradients in deep networks.  

3. **B** - Preserves the spatial dimensions of the input.  
   - Explanation: Padding ensures the output dimensions remain equal to the input dimensions.  

4. **C** - Large differences between predicted and actual values.  
   - Explanation: High MSE indicates significant prediction errors.  

5. **B** - To handle objects of different sizes and shapes.  
   - Explanation: Anchor boxes allow models to detect objects at multiple scales and aspect ratios.  

6. **A** - Early layers are frozen, and later layers are fine-tuned.  
   - Explanation: Early layers capture general features, while later layers are task-specific.  

7. **B** - F1-Score.  
   - Explanation: F1-Score balances precision and recall, making it suitable for imbalanced datasets.  

8. **B** - To reduce computational complexity and extract dominant features.  
   - Explanation: Pooling layers downsample feature maps while retaining critical information.  

9. **A** - L1 Regularization.  
   - Explanation: L1 regularization penalizes non-zero weights, promoting sparsity.  

10. **B** - \(63 \times 63 \times 64\).  
    - Explanation: The formula for output size is \((W - K)/S + 1\). Here, \((128 - 3)/2 + 1 = 63\).  

---

#### **Part B: Short-Answer Questions**

11. **Bottleneck in Autoencoder:**  
    - The bottleneck compresses input data into a latent representation, reducing dimensionality while retaining essential features. This aids in efficient reconstruction and feature extraction.  

12. **Intersection Over Union (IoU):**  
    - IoU measures the overlap between predicted and ground truth bounding boxes. It ensures accurate localization and is critical for evaluating object detection performance.  

13. **Feature Extraction vs. Fine-Tuning:**  
    - Feature extraction uses pre-trained layers without modifying their weights, while fine-tuning updates specific layers to adapt to the new task.  

14. **ReLU vs. Sigmoid:**  
    - ReLU avoids the vanishing gradient problem by outputting zero for negative inputs and linear values for positive inputs, ensuring efficient training.  

15. **Non-Max Suppression:**  
    - Non-max suppression removes redundant bounding boxes, retaining only the most confident predictions, improving object detection accuracy.  

---

This question paper provides another robust practice set. Let me know if you need further variations!

