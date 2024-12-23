### Machine Learning and Computer Vision Questions (Set 2)

#### Model Evaluation and Metrics
1. **What does a high F1-score indicate in a classification problem?**  
   - A) High precision and low recall  
   - B) High recall and low precision  
   - C) A balance between precision and recall  
   - D) Only high accuracy  
   **Answer:** C  
   **Explanation:** The F1-score is the harmonic mean of precision and recall, reflecting their balance.

2. **What metric is most appropriate for evaluating a model on an imbalanced dataset?**  
   - A) Accuracy  
   - B) Precision  
   - C) Recall  
   - D) F1-Score  
   **Answer:** D  
   **Explanation:** F1-Score handles imbalanced datasets by balancing precision and recall.

3. **Which of the following can be used to evaluate a regression model?**  
   - A) ROC Curve  
   - B) Mean Absolute Error (MAE)  
   - C) F1-Score  
   - D) Confusion Matrix  
   **Answer:** B  
   **Explanation:** MAE measures the average absolute difference between predicted and actual values.

4. **How does precision differ from recall?**  
   - A) Precision measures positive predictions, while recall measures negative predictions.  
   - B) Precision measures the proportion of correct positive predictions, while recall measures the proportion of actual positives identified.  
   - C) Precision is specific to multi-class problems, while recall is for binary classification.  
   - D) Precision is used for regression tasks, while recall is for classification tasks.  
   **Answer:** B  
   **Explanation:** Precision and recall focus on different aspects of true positives.

#### Neural Network Concepts
5. **What is the purpose of using batch normalization in neural networks?**  
   - A) To reduce overfitting  
   - B) To speed up convergence and stabilize training  
   - C) To increase model complexity  
   - D) To simplify feature maps  
   **Answer:** B  
   **Explanation:** Batch normalization normalizes activations to accelerate training and improve stability.

6. **Which activation function is most suitable for binary classification problems?**  
   - A) ReLU  
   - B) Sigmoid  
   - C) Tanh  
   - D) Softmax  
   **Answer:** B  
   **Explanation:** Sigmoid maps outputs to a probability range between 0 and 1, suitable for binary classification.

7. **What is the vanishing gradient problem in neural networks?**  
   - A) Gradients become too large and lead to unstable training.  
   - B) Gradients become too small, making weight updates negligible.  
   - C) The network converges too quickly.  
   - D) The network fails to initialize weights properly.  
   **Answer:** B  
   **Explanation:** The vanishing gradient problem hampers deep network training as gradients diminish during backpropagation.

8. **Which neural network layer is typically used for reducing spatial dimensions in feature maps?**  
   - A) Fully connected layer  
   - B) Convolutional layer  
   - C) Pooling layer  
   - D) Activation layer  
   **Answer:** C  
   **Explanation:** Pooling layers downsample feature maps, reducing spatial dimensions.

#### Transfer Learning
9. **What is the primary advantage of freezing pre-trained layers in transfer learning?**  
   - A) To improve computational efficiency during training.  
   - B) To allow all layers to adapt to the new task.  
   - C) To improve performance on large datasets.  
   - D) To reduce the number of parameters.  
   **Answer:** A  
   **Explanation:** Freezing layers reduces training time by reusing pre-trained weights.

10. **Which dataset characteristic makes transfer learning less effective?**  
    - A) Small dataset size  
    - B) High similarity with the source dataset  
    - C) High dissimilarity with the source dataset  
    - D) Balanced dataset classes  
    **Answer:** C  
    **Explanation:** Transfer learning works best when the source and target datasets are similar.

#### Autoencoders
11. **What is the main difference between variational autoencoders and standard autoencoders?**  
    - A) Variational autoencoders use non-linear activation functions.  
    - B) Variational autoencoders learn a probability distribution over the data.  
    - C) Standard autoencoders cannot be used for reconstruction.  
    - D) Variational autoencoders do not use latent representations.  
    **Answer:** B  
    **Explanation:** Variational autoencoders learn probabilistic distributions for generative modeling.

12. **What happens in the decoder part of an autoencoder?**  
    - A) Data is compressed into lower dimensions.  
    - B) Data is reconstructed to match the original input.  
    - C) Latent representations are clustered.  
    - D) Data augmentation is performed.  
    **Answer:** B  
    **Explanation:** The decoder reconstructs the original input from the compressed latent representation.

#### Image Classification
13. **What is the main purpose of using dropout layers in CNNs?**  
    - A) To reduce computational cost  
    - B) To prevent overfitting by randomly dropping neurons  
    - C) To increase the number of feature maps  
    - D) To improve batch normalization  
    **Answer:** B  
    **Explanation:** Dropout regularizes the network by randomly dropping units during training.

14. **Which metric is best for evaluating a multi-label classification problem?**  
    - A) Accuracy  
    - B) Precision and Recall  
    - C) Mean Absolute Error  
    - D) ROC-AUC  
    **Answer:** B  
    **Explanation:** Precision and recall are effective for evaluating multi-label problems.

#### Object Detection and Segmentation
15. **What is the primary purpose of non-max suppression in object detection?**  
    - A) To enhance feature extraction  
    - B) To merge overlapping bounding boxes  
    - C) To discard redundant bounding boxes  
    - D) To increase model confidence  
    **Answer:** C  
    **Explanation:** Non-max suppression removes redundant boxes, retaining only the most confident predictions.

16. **How does semantic segmentation differ from object detection?**  
    - A) Semantic segmentation focuses on edge detection.  
    - B) Object detection identifies pixels, while semantic segmentation identifies objects.  
    - C) Semantic segmentation assigns labels to every pixel, while object detection localizes objects with bounding boxes.  
    - D) They are identical in functionality.  
    **Answer:** C  
    **Explanation:** Semantic segmentation classifies every pixel, whereas object detection provides bounding boxes.

#### Regularization
17. **What is the primary goal of early stopping?**  
    - A) To accelerate convergence.  
    - B) To prevent overfitting by halting training when validation performance stops improving.  
    - C) To ensure all layers are fully trained.  
    - D) To reset weights periodically during training.  
    **Answer:** B  
    **Explanation:** Early stopping halts training when further improvements on validation loss are not observed.

18. **What type of regularization encourages sparsity in model weights?**  
    - A) Dropout  
    - B) L1 Regularization  
    - C) L2 Regularization  
    - D) Batch Normalization  
    **Answer:** B  
    **Explanation:** L1 regularization penalizes non-zero weights, encouraging sparsity.

19. **How does increasing the regularization parameter affect a model?**  
    - A) Increases the model's complexity  
    - B) Reduces the model's complexity  
    - C) Improves training speed  
    - D) Decreases generalization  
    **Answer:** B  
    **Explanation:** Stronger regularization reduces complexity by penalizing larger weights.

#### Miscellaneous
20. **Why is data augmentation important in image classification?**  
    - A) To increase dataset size artificially  
    - B) To reduce the need for pre-trained models  
    - C) To improve computational efficiency  
    - D) To simplify model architecture  
    **Answer:** A  
    **Explanation:** Data augmentation generates variations of existing images, improving the model's robustness.

21. **What does a fully connected layer in a CNN do?**  
    - A) Reduces feature map size  
    - B) Combines extracted features to make predictions  
    - C) Performs spatial convolutions  
    - D) Normalizes data  
    **Answer:** B  
    **Explanation:** Fully connected layers aggregate features to produce final predictions.

22. **Which of the following optimizers adapts learning rates for each parameter?**  
    - A) SGD  
    - B) Adam  
    - C) RMSprop  
    - D) Momentum  
    **Answer:** B  
    **Explanation:** Adam uses adaptive learning rates for faster and stable optimization.

23. **What is the effect of using a smaller batch size during training?**  
    - A) Faster convergence  
    - B) Noisy gradient updates  
    - C) Reduced overfitting  
    - D) Increased generalization  
    **Answer:** B  
    **Explanation:** Smaller batches produce noisier gradients but may help escape local minima.

24. **Which loss function is suitable for a binary classification problem?**  
    - A) Mean Squared Error  
    - B) Binary Cross-Entropy  
    - C) Hinge Loss  
    - D) Categorical Cross-Entropy  
    **Answer:** B  
    **Explanation:** Binary cross-entropy measures the difference between predicted and actual probabilities for binary outcomes.

25. **How does increasing filter size in a CNN affect feature extraction?**  
    - A) Detects finer details  
    - B) Captures broader patterns  
    - C) Increases overfitting risk  
    - D) Reduces spatial dimensions  
    **Answer:** B  
    **Explanation:** Larger filters detect more global features, while smaller filters capture finer details.  

Let me know if you'd like further customization or expansion!
