Here’s a new set of 25 multiple-choice questions (MCQs) based on topics like model evaluation, loss functions, transfer learning, and neural networks, tailored to the style of the attached document:

---

### Model Evaluation and Metrics

1. **During evaluation, you notice the model performs well on the training set but poorly on the validation set. What is the likely issue?**  
   - A) Underfitting  
   - B) Overfitting  
   - C) Balanced bias and variance  
   - D) Data leakage  
   **Answer:** B  

2. **What is the main advantage of using a confusion matrix?**  
   - A) It computes the model’s loss.  
   - B) It visualizes true positives, false positives, true negatives, and false negatives.  
   - C) It calculates mean squared error.  
   - D) It identifies overfitting.  
   **Answer:** B  

3. **What metric is most useful for evaluating an imbalanced binary classification problem?**  
   - A) Accuracy  
   - B) F1-Score  
   - C) Mean Squared Error  
   - D) ROC-AUC  
   **Answer:** B  

4. **Which method can help avoid data leakage during model training?**  
   - A) Performing cross-validation  
   - B) Using the same dataset for training and testing  
   - C) Including target labels in the feature set  
   - D) Normalizing data after splitting  
   **Answer:** A  

5. **In a binary classification task, the recall is low. What does this indicate?**  
   - A) Many false negatives  
   - B) Many false positives  
   - C) High precision  
   - D) High accuracy  
   **Answer:** A  

---

### Loss Functions

6. **Which loss function is suitable for multi-class classification problems?**  
   - A) Mean Squared Error  
   - B) Binary Cross-Entropy  
   - C) Categorical Cross-Entropy  
   - D) Hinge Loss  
   **Answer:** C  

7. **What is the primary advantage of using Huber loss for regression?**  
   - A) It is less sensitive to outliers.  
   - B) It works only for binary outcomes.  
   - C) It penalizes false negatives heavily.  
   - D) It is computationally faster than MSE.  
   **Answer:** A  

8. **In binary classification, what does binary cross-entropy measure?**  
   - A) Squared difference between actual and predicted values.  
   - B) Logarithmic loss for class probabilities.  
   - C) Difference in predicted labels.  
   - D) Precision of predictions.  
   **Answer:** B  

9. **Why is hinge loss used in support vector machines?**  
   - A) To calculate probabilities.  
   - B) To penalize predictions that fall within the margin.  
   - C) To improve recall.  
   - D) To avoid underfitting.  
   **Answer:** B  

10. **Which property makes Mean Absolute Error (MAE) unique compared to Mean Squared Error (MSE)?**  
    - A) MAE is less sensitive to outliers.  
    - B) MAE penalizes larger errors more heavily.  
    - C) MAE measures accuracy instead of error.  
    - D) MAE works for classification tasks.  
    **Answer:** A  

---

### Neural Networks

11. **What is the main role of activation functions in neural networks?**  
    - A) To initialize weights  
    - B) To introduce non-linearity  
    - C) To prevent overfitting  
    - D) To normalize data  
    **Answer:** B  

12. **What does a fully connected layer in a neural network do?**  
    - A) Normalizes input data  
    - B) Extracts spatial features  
    - C) Combines features for classification  
    - D) Reduces the number of parameters  
    **Answer:** C  

13. **Which activation function is most likely to suffer from the vanishing gradient problem?**  
    - A) Sigmoid  
    - B) ReLU  
    - C) Softmax  
    - D) Leaky ReLU  
    **Answer:** A  

14. **What is the effect of using a pooling layer in a CNN?**  
    - A) It increases spatial resolution.  
    - B) It reduces spatial dimensions while retaining important features.  
    - C) It reduces the number of channels.  
    - D) It performs data augmentation.  
    **Answer:** B  

15. **Why are residual connections used in deep neural networks?**  
    - A) To reduce model complexity  
    - B) To improve gradient flow during backpropagation  
    - C) To replace activation functions  
    - D) To reduce computation  
    **Answer:** B  

---

### Transfer Learning

16. **What is the primary benefit of transfer learning?**  
    - A) It requires more training time.  
    - B) It adapts pre-trained features to a new task.  
    - C) It eliminates the need for validation data.  
    - D) It works only with large datasets.  
    **Answer:** B  

17. **What is typically fine-tuned in a pre-trained model during transfer learning?**  
    - A) All layers equally  
    - B) Output layers for task-specific adaptation  
    - C) Only convolutional layers  
    - D) Input layers  
    **Answer:** B  

18. **Which pre-trained model is commonly used for image classification tasks?**  
    - A) ResNet  
    - B) LSTM  
    - C) RNN  
    - D) Naive Bayes  
    **Answer:** A  

19. **When is transfer learning least effective?**  
    - A) When the target task is similar to the source task  
    - B) When the source and target domains are significantly different  
    - C) When the dataset size is small  
    - D) When using pre-trained weights  
    **Answer:** B  

20. **Why are earlier layers often frozen during transfer learning?**  
    - A) They contain task-specific features.  
    - B) They contain generic features suitable for many tasks.  
    - C) They reduce computational efficiency.  
    - D) They cause overfitting.  
    **Answer:** B  

---

### Computer Vision and Object Detection

21. **Which metric is used to measure overlap between predicted and ground truth bounding boxes in object detection?**  
    - A) F1-Score  
    - B) Mean Squared Error  
    - C) Intersection Over Union (IoU)  
    - D) Cross-Entropy  
    **Answer:** C  

22. **What is the role of non-max suppression in object detection?**  
    - A) To increase IoU  
    - B) To merge overlapping bounding boxes  
    - C) To remove redundant predictions and retain the most confident ones  
    - D) To balance precision and recall  
    **Answer:** C  

23. **How does semantic segmentation differ from instance segmentation?**  
    - A) Semantic segmentation identifies individual object instances.  
    - B) Instance segmentation classifies every pixel into categories.  
    - C) Semantic segmentation assigns labels to pixels without distinguishing instances.  
    - D) They are the same.  
    **Answer:** C  

24. **What is the purpose of anchor boxes in object detection models?**  
    - A) To identify edges in an image  
    - B) To handle objects of different scales and aspect ratios  
    - C) To normalize bounding boxes  
    - D) To segment objects in an image  
    **Answer:** B  

25. **What is the function of a Flatten layer in CNNs?**  
    - A) To combine feature maps into a 1D vector  
    - B) To increase the number of channels  
    - C) To reduce spatial dimensions  
    - D) To act as a pooling layer  
    **Answer:** A  

---

These questions mimic the style of the original document and are designed for quiz preparation. Let me know if you'd like more or specific adjustments!
