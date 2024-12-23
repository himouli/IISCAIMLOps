Here’s another set of 25 practice questions with detailed answers and explanations, continuing the style of scenario-based and parameter-specific questions for convolutional neural networks, transfer learning, and metrics.

---

### ConvNet Architecture and Parameters

1. **A ConvNet layer has the following configuration:**  
   Input size = \(64 \times 64 \times 3\), Filter size = \(3 \times 3\), Number of filters = 32, Stride = 1, Padding = 'valid'.  
   What is the output size?  
   - A) \(62 \times 62 \times 32\)  
   - B) \(64 \times 64 \times 32\)  
   - C) \(30 \times 30 \times 32\)  
   - D) \(31 \times 31 \times 32\)  
   **Answer:** A  
   **Explanation:** The formula for output size is \((W - K) / S + 1\). Here, \(W = 64\), \(K = 3\), \(S = 1\), so \((64 - 3)/1 + 1 = 62\). Depth is determined by the number of filters.

2. **For an input size of \(128 \times 128 \times 3\), a ConvNet applies the following layers:**  
   - **Layer 1:** Conv2D, \(5 \times 5\) kernel, stride = 2, filters = 16, padding = 'same'.  
   - **Layer 2:** MaxPooling, \(2 \times 2\) pool size, stride = 2.  
   What is the output size after Layer 2?  
   - A) \(32 \times 32 \times 16\)  
   - B) \(64 \times 64 \times 16\)  
   - C) \(16 \times 16 \times 16\)  
   - D) \(128 \times 128 \times 16\)  
   **Answer:** A  
   **Explanation:**  
   - After Layer 1: \(128 / 2 = 64 \times 64 \times 16\).  
   - After Layer 2: \(64 / 2 = 32 \times 32 \times 16\).

3. **A CNN has the following setup:**  
   **Input:** \(256 \times 256 \times 3\),  
   **Layer 1:** Conv2D, \(3 \times 3\), 32 filters, stride = 1, padding = 'valid'.  
   **Layer 2:** Conv2D, \(5 \times 5\), 64 filters, stride = 2, padding = 'valid'.  
   What is the output size after Layer 2?  
   - A) \(125 \times 125 \times 64\)  
   - B) \(252 \times 252 \times 64\)  
   - C) \(62 \times 62 \times 64\)  
   - D) \(124 \times 124 \times 64\)  
   **Answer:** A  
   **Explanation:**  
   - Layer 1: \((256 - 3) + 1 = 254 \times 254 \times 32\).  
   - Layer 2: \((254 - 5)/2 + 1 = 125 \times 125 \times 64\).

4. **In a ConvNet, what is the primary effect of using larger kernel sizes (e.g., \(7 \times 7\) instead of \(3 \times 3\))?**  
   - A) Increased spatial detail detection  
   - B) Reduced parameter count  
   - C) Broader feature detection  
   - D) Higher computational efficiency  
   **Answer:** C  
   **Explanation:** Larger kernels capture broader patterns but require more computations.

5. **A \(39 \times 39 \times 3\) RGB image passes through the following ConvNet layers:**  
   - **Layer 1:** Conv2D, \(3 \times 3\), stride = 1, 10 filters, padding = 'valid'.  
   - **Layer 2:** Conv2D, \(5 \times 5\), stride = 2, 20 filters, padding = 'valid'.  
   - **Layer 3:** MaxPooling, \(2 \times 2\), stride = 2.  
   What is the output size after Layer 3?  
   - A) \(8 \times 8 \times 20\)  
   - B) \(9 \times 9 \times 20\)  
   - C) \(19 \times 19 \times 20\)  
   - D) \(18 \times 18 \times 20\)  
   **Answer:** A  
   **Explanation:**  
   - Layer 1: \((39 - 3)/1 + 1 = 37 \times 37 \times 10\).  
   - Layer 2: \((37 - 5)/2 + 1 = 17 \times 17 \times 20\).  
   - Layer 3: \(17 / 2 = 8 \times 8 \times 20\).

---

### Transfer Learning

6. **Which pre-trained model is commonly used for object detection tasks?**  
   - A) ResNet  
   - B) YOLO  
   - C) Inception  
   - D) VGG  
   **Answer:** B  
   **Explanation:** YOLO (You Only Look Once) is widely used for object detection due to its speed and accuracy.

7. **What is the primary reason for freezing layers in a pre-trained model?**  
   - A) To fine-tune task-specific features  
   - B) To retain generic features learned during pre-training  
   - C) To reduce model complexity  
   - D) To avoid vanishing gradients  
   **Answer:** B  

8. **You use transfer learning for a dataset with 1000 images. Which layers of the pre-trained model should be fine-tuned for optimal performance?**  
   - A) All layers  
   - B) Early convolutional layers  
   - C) Fully connected layers  
   - D) Middle layers  
   **Answer:** C  

---

### Regularization and Optimization

9. **What is the primary effect of adding dropout layers in a neural network?**  
   - A) Reduces vanishing gradients  
   - B) Prevents overfitting  
   - C) Improves computational efficiency  
   - D) Increases training time  
   **Answer:** B  

10. **L2 regularization penalizes which of the following?**  
    - A) Sparse weights  
    - B) Large weights  
    - C) Small weights  
    - D) Weight gradients  
    **Answer:** B  

11. **How does early stopping prevent overfitting?**  
    - A) By reducing the model complexity  
    - B) By stopping training when validation performance degrades  
    - C) By increasing dropout rates  
    - D) By penalizing large weights  
    **Answer:** B  

---

### Object Detection and Segmentation

12. **In object detection, what is the significance of Intersection Over Union (IoU)?**  
    - A) Measures classification accuracy  
    - B) Evaluates the overlap between predicted and ground truth boxes  
    - C) Increases training speed  
    - D) Reduces computational cost  
    **Answer:** B  

13. **What is the main role of non-max suppression?**  
    - A) To adjust anchor box sizes  
    - B) To merge overlapping bounding boxes  
    - C) To remove redundant predictions  
    - D) To increase the IoU score  
    **Answer:** C  

14. **In semantic segmentation, what does each pixel in the image represent?**  
    - A) A bounding box  
    - B) A class label  
    - C) A feature map  
    - D) A latent variable  
    **Answer:** B  

15. **What is the primary goal of anchor boxes in object detection?**  
    - A) To identify edges  
    - B) To improve recall  
    - C) To handle objects of varying sizes and shapes  
    - D) To reduce false positives  
    **Answer:** C  

---

### Autoencoders

16. **What is the bottleneck layer in an autoencoder responsible for?**  
    - A) Classifying data  
    - B) Compressing input into a latent representation  
    - C) Upsampling feature maps  
    - D) Increasing the number of features  
    **Answer:** B  

17. **Which application is most suited for autoencoders?**  
    - A) Object detection  
    - B) Anomaly detection  
    - C) Regression tasks  
    - D) Multi-class classification  
    **Answer:** B  

---

### Additional Topics

18. **What is the primary benefit of using batch normalization in deep networks?**  
    - A) Reduces overfitting  
    - B) Speeds up convergence by normalizing layer inputs  
    - C) Prevents vanishing gradients  
    - D) Reduces computational cost  
    **Answer:** B  

19. **Which metric is best for evaluating imbalanced datasets?**  
    - A) Accuracy  
    - B) Precision-Recall Curve  
    - C) Mean Squared Error  
    - D) ROC Curve  
    **Answer:** B  

20. **What is the effect of a larger stride in convolution layers?**  
    - A) Reduces spatial dimensions of output  
    - B) Increases the depth of output  
    - C) Retains spatial dimensions  
    - D) Increases computation time  
    **Answer:** A  

21. **What does padding 'same' ensure in convolution operations?**  
    - A) Output size is the same as input size  
    - B) Output size is reduced  
    - C) Padding size is dynamic  
    - D) Filters overlap completely  
    **Answer:** A  

22. **How does data augmentation improve model training?**  
    - A) Reduces overfitting by artificially expanding the training set  
    - B) Speeds up convergence  
    - C) Increases dropout rates  
    - D) Simplifies feature extraction  
    **Answer:** A  

23. **What type of loss function is suitable for binary classification tasks?**  
    - A) Hinge loss  
    - B) Binary cross-entropy  
    - C) Mean Absolute Error  
    - D) Categorical cross-entropy  
    **Answer:** B  

24. **What is the role of a Flatten layer in CNNs?**  
    - A) Converts feature maps into a vector  
    - B) Reduces spatial dimensions  
    - C) Normalizes activations  
    - D) Adds dropout  
    **Answer:** A  

25. **Why are global average pooling layers used in modern CNNs?**  
    - A) To reduce overfitting by removing fully connected layers  
    - B) To replace convolutional layers  
    - C) To increase spatial resolution  
    - D) To speed up training  
    **Answer:** A  

---

Let me know if you’d like additional sets or refinements!
