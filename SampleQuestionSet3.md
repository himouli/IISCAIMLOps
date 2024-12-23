### Machine Learning and Computer Vision Questions (Based on the Sample)

#### Model Evaluation and Metrics
1. **Which of the following indicates underfitting in a model?**  
   - A) Low training error and low validation error  
   - B) High training error and low validation error  
   - C) High training error and high validation error  
   - D) Low training error and high validation error  
   **Answer:** C  
   **Explanation:** Underfitting occurs when the model fails to learn sufficient patterns from the data, leading to high errors across training and validation sets.

2. **What metric would you use to evaluate the performance of a multi-class classifier?**  
   - A) Mean Absolute Error  
   - B) Accuracy and F1-Score  
   - C) ROC-AUC  
   - D) Mean Squared Error  
   **Answer:** B  
   **Explanation:** Accuracy and F1-Score are ideal for evaluating multi-class classification, with F1-Score balancing precision and recall.

3. **In a confusion matrix, what does the term 'false positive' represent?**  
   - A) Correctly predicted positive instance  
   - B) Incorrectly predicted positive instance  
   - C) Correctly predicted negative instance  
   - D) Incorrectly predicted negative instance  
   **Answer:** B  
   **Explanation:** A false positive is when the model predicts a positive outcome incorrectly.

#### Neural Network Concepts
4. **What is the primary purpose of the ReLU activation function in neural networks?**  
   - A) To normalize outputs  
   - B) To introduce non-linearity  
   - C) To reduce overfitting  
   - D) To optimize training speed  
   **Answer:** B  
   **Explanation:** ReLU introduces non-linearity, allowing networks to learn complex patterns.

5. **Why are pooling layers used in CNNs?**  
   - A) To increase feature map size  
   - B) To reduce computational complexity and retain important features  
   - C) To replace convolutional layers  
   - D) To fully connect layers  
   **Answer:** B  
   **Explanation:** Pooling layers downsample feature maps, reducing dimensionality while retaining important patterns.

#### Transfer Learning
6. **Which of the following is a benefit of transfer learning?**  
   - A) Requires large amounts of labeled data  
   - B) Improves training speed and reduces computational cost  
   - C) Discards pre-trained weights  
   - D) Requires extensive re-training  
   **Answer:** B  
   **Explanation:** Transfer learning leverages pre-trained models, saving time and resources.

7. **Which part of a pre-trained model is commonly fine-tuned for a new classification task?**  
   - A) Input layers  
   - B) Middle layers  
   - C) Output layers  
   - D) All layers equally  
   **Answer:** C  
   **Explanation:** Output layers are adjusted to match the new task's output classes.

#### Autoencoders
8. **What is the role of the bottleneck in an autoencoder?**  
   - A) To increase dimensionality  
   - B) To compress and represent essential features  
   - C) To reconstruct input without loss  
   - D) To normalize inputs  
   **Answer:** B  
   **Explanation:** The bottleneck compresses data into a lower-dimensional representation, preserving essential features.

9. **Which task is best suited for autoencoders?**  
   - A) Classification  
   - B) Anomaly detection  
   - C) Regression  
   - D) Object detection  
   **Answer:** B  
   **Explanation:** Autoencoders excel at detecting anomalies by reconstructing patterns and identifying outliers.

#### Image Classification
10. **What is the purpose of a softmax activation function in the output layer of a neural network?**  
    - A) To perform binary classification  
    - B) To normalize probabilities across multiple classes  
    - C) To increase feature map size  
    - D) To detect edges  
    **Answer:** B  
    **Explanation:** Softmax outputs normalized probabilities, ensuring their sum equals 1.

11. **Which of the following methods helps reduce overfitting in a deep learning model?**  
    - A) Increasing batch size  
    - B) Using dropout layers  
    - C) Increasing learning rate  
    - D) Decreasing regularization strength  
    **Answer:** B  
    **Explanation:** Dropout randomly deactivates neurons, preventing the model from over-relying on specific pathways.

#### Object Detection and Segmentation
12. **What is Intersection Over Union (IoU) used for in object detection?**  
    - A) Classifying objects  
    - B) Measuring the overlap between predicted and ground truth boxes  
    - C) Detecting small objects  
    - D) Evaluating segmentation maps  
    **Answer:** B  
    **Explanation:** IoU measures the accuracy of bounding box predictions.

13. **What does a high IoU value indicate?**  
    - A) Poor overlap  
    - B) Accurate object localization  
    - C) Incorrect predictions  
    - D) Reduced confidence  
    **Answer:** B  
    **Explanation:** High IoU reflects a close match between predicted and true bounding boxes.

#### Regularization
14. **What is the purpose of L2 regularization?**  
    - A) To enforce sparsity  
    - B) To penalize large weights  
    - C) To reduce dimensionality  
    - D) To increase training time  
    **Answer:** B  
    **Explanation:** L2 regularization discourages large weights, reducing overfitting.

15. **How does dropout improve model performance?**  
    - A) By preventing co-adaptation of neurons  
    - B) By increasing the number of parameters  
    - C) By normalizing activations  
    - D) By using smaller datasets  
    **Answer:** A  
    **Explanation:** Dropout ensures the network learns robust features by randomly deactivating neurons.

---

Would you like me to expand further or focus on specific subtopics?
