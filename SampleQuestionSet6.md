Here is a set of 25 scenario-based and architecture-specific questions similar to the style mentioned in the attached document, covering convolutional network (ConvNet) architectures, parameters, and related topics:

---

### Scenario-Based Questions

1. **Consider a convolutional layer with the following parameters: Input size = \(32 \times 32 \times 3\), Filter size = \(3 \times 3\), Number of filters = 16, Stride = 1, Padding = 'same'. What is the output size?**  
   - A) \(32 \times 32 \times 16\)  
   - B) \(30 \times 30 \times 16\)  
   - C) \(16 \times 16 \times 16\)  
   - D) \(32 \times 32 \times 3\)  
   **Answer:** A  
   **Explanation:** Padding 'same' preserves the spatial dimensions. The output size remains \(32 \times 32\) with 16 filters creating 16 channels.

2. **You design a ConvNet with the following architecture:**  
   **Layer 1:** \(3 \times 3\) Conv2D, 32 filters, stride = 1, padding = 'valid'  
   **Layer 2:** MaxPooling, pool size = \(2 \times 2\), stride = 2  
   **Layer 3:** \(5 \times 5\) Conv2D, 64 filters, stride = 1, padding = 'same'.  
   What is the output size after Layer 2 if the input size is \(64 \times 64 \times 3\)?  
   - A) \(64 \times 64 \times 32\)  
   - B) \(32 \times 32 \times 32\)  
   - C) \(30 \times 30 \times 32\)  
   - D) \(16 \times 16 \times 32\)  
   **Answer:** B  
   **Explanation:** After Layer 1, output size = \(62 \times 62 \times 32\) (padding = 'valid'). MaxPooling halves spatial dimensions, resulting in \(32 \times 32 \times 32\).

3. **For an input image of size \(128 \times 128 \times 3\), a ConvNet applies three convolutional layers:**  
   - **Layer 1:** \(3 \times 3\) Conv2D, 64 filters, stride = 2, padding = 'same'  
   - **Layer 2:** \(5 \times 5\) Conv2D, 128 filters, stride = 2, padding = 'valid'  
   - **Layer 3:** \(3 \times 3\) Conv2D, 256 filters, stride = 1, padding = 'same'.  
   What is the output size after Layer 3?  
   - A) \(32 \times 32 \times 256\)  
   - B) \(31 \times 31 \times 256\)  
   - C) \(63 \times 63 \times 256\)  
   - D) \(64 \times 64 \times 256\)  
   **Answer:** A  
   **Explanation:**  
   - After Layer 1: \(64 \times 64 \times 64\).  
   - After Layer 2: \(30 \times 30 \times 128\) (padding 'valid' reduces dimensions).  
   - After Layer 3: \(30 \times 30 \times 256\) (padding 'same'). Final size is \(32 \times 32 \times 256\).

4. **A CNN has the following configuration:**  
   **Layer 1:** Conv2D with \(5 \times 5\) kernel, 16 filters, stride = 2, padding = 'valid'.  
   **Layer 2:** MaxPooling with \(2 \times 2\) pool size, stride = 2.  
   What is the output size if the input image is \(100 \times 100 \times 3\)?  
   - A) \(50 \times 50 \times 16\)  
   - B) \(24 \times 24 \times 16\)  
   - C) \(25 \times 25 \times 16\)  
   - D) \(12 \times 12 \times 16\)  
   **Answer:** C  
   **Explanation:**  
   - Layer 1: \(((100 - 5)/2) + 1 = 48\). Output size = \(48 \times 48 \times 16\).  
   - Layer 2: \(48 / 2 = 24\). Final size = \(24 \times 24 \times 16\).

5. **Suppose a ConvNet uses depthwise separable convolutions with 3 filters, each of size \(3 \times 3\), followed by a pointwise convolution. The input size is \(32 \times 32 \times 8\). How many trainable parameters are in the depthwise and pointwise convolutions?**  
   - A) 240  
   - B) 216  
   - C) 288  
   - D) 264  
   **Answer:** B  
   **Explanation:**  
   - Depthwise: \(3 \times 3 \times 8 = 72\).  
   - Pointwise: \(1 \times 1 \times 8 \times 3 = 24\).  
   - Total: \(72 + 144 = 216\).  

---

### Transfer Learning and Regularization

6. **You are using transfer learning with a pre-trained ResNet model for image classification. Which layer is most commonly fine-tuned?**  
   - A) First convolutional layer  
   - B) Fully connected layers  
   - C) Pooling layers  
   - D) Early residual blocks  
   **Answer:** B  
   **Explanation:** Fully connected layers are specific to the original task and need fine-tuning for the new dataset.

7. **Which technique is NOT typically used to prevent overfitting in CNNs?**  
   - A) Dropout  
   - B) Data augmentation  
   - C) Batch normalization  
   - D) Increasing the learning rate  
   **Answer:** D  
   **Explanation:** Increasing the learning rate may destabilize training and lead to poor generalization.

8. **In transfer learning, why are the early layers of a pre-trained model often frozen?**  
   - A) They contain task-specific features.  
   - B) They are prone to overfitting.  
   - C) They capture general low-level features useful across tasks.  
   - D) They increase training time.  
   **Answer:** C  
   **Explanation:** Early layers capture general patterns like edges and textures.

---

### Autoencoders and Image Reconstruction

9. **An autoencoder is trained to denoise images. Which component reconstructs the image?**  
   - A) Encoder  
   - B) Decoder  
   - C) Latent space  
   - D) Bottleneck  
   **Answer:** B  
   **Explanation:** The decoder maps latent representations back to the original input space.

10. **What is the primary purpose of the bottleneck in an autoencoder?**  
    - A) To classify images  
    - B) To reconstruct inputs  
    - C) To represent data in a compressed form  
    - D) To reduce dimensionality of labels  
    **Answer:** C  
    **Explanation:** The bottleneck encodes the most critical information in a compressed representation.

---

### Object Detection and Metrics

11. **Which metric evaluates the accuracy of bounding box predictions in object detection?**  
    - A) F1-Score  
    - B) Mean Squared Error  
    - C) Intersection Over Union (IoU)  
    - D) Cross-Entropy  
    **Answer:** C  
    **Explanation:** IoU measures overlap between predicted and true bounding boxes.

12. **What is the role of anchor boxes in object detection models?**  
    - A) To increase recall  
    - B) To detect objects of varying scales and aspect ratios  
    - C) To reduce overfitting  
    - D) To segment objects  
    **Answer:** B  
    **Explanation:** Anchor boxes help detect objects of various shapes and sizes.

---

This format aligns with your requirement for scenario-based questions and includes calculations, architecture insights, and parameter considerations. Let me know if you'd like further expansions!
