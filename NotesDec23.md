



mportant: Add layers of same shape !

Q: Can we add layers of shapes (30,30,64) and (30,30,32)?

A: No

Q: But what happens if you have a pooling layer in between?

A: Spatial Dimension reduction due to stride

Solution: Use Strides in the Conv layer in the skip connection.



With residual connections, you can build networks of arbitrary depth, without having to worry about vanishing gradients. We will see an example later.

Intuitions on why residual blocks work:

    Shorter path for gradients
    Easy to learn the identity matrix
    Ensemble of shallow networks







Q: How many parameters does the BN layer introduce in the above model?

A: 128

Q: HOW?

A: Batch Normalization layer introduces four parameters per channel. However, only two parameters, γ and β, are learnable/trainable parameters used to apply scaling and shifting to the transformation. The remaining two parameters, moving_mean and moving_variance, are non-trainable and are directly calculated from the mean across the batch and saved as part of the state of the Batch Normalization layer.

Here, the number of channels in the preceding layer is 32. Hence, the total number of parameters is equal to 32 * 4 = 128, but out of this, only 128/2 = 64 are trainable.



Intuitions:

    Batch Normalization is also a (weak) regularization method.
        increases no. of params
        but also adds noise ~ data augmentation ~ dropout


Depthwise separable convolutions

    This layer performs a spatial convolution [Depthwise Conv.] on each channel of its input, independently, before mixing output channels via a pointwise convolution.

    Depthwise separable convolution = Depthwise Conv. + Pointwise Conv.

    This makes your model smaller and acts as a strong prior. We impose a strong prior by assuming that spatial patterns and cross-channel patterns can be modeled separately.This is equivalent to separating the learning of spatial features and the learning of channel-wise features.

    Depthwise separable convolution relies on the assumption that spatial locations in intermediate activations are highly correlated, but different channels are highly independent. So we never use depthwise separable convolution after the input layer. Because RGB channels are highly correlated.



# Q: Verify the number of parameters in the separable_conv2D layer.
# A: Explanation of the parameter calculation for the SeparableConv2D layer:
#    - Depthwise convolution: (32 filters * 3x3 kernel) = 32 * (3*3) = 288 parameters (for each channel).
#    - Pointwise convolution: (32 input channels * 1x1 kernel * 64 output filters) = 32 * 1 * 1 * 64 = 2048 parameters.
#    - Biases: Each output filter has a bias term, so the number of biases is 64.
#    Total parameters: 288 (depthwise) + 2048 (pointwise) + 64 (biases) = 2400 parameters.
# Note: In Keras' SeparableConv2D, the bias in the depthwise convolution is not included, which is why it's excluded in the first calculation.

# Considering all biases:
#    Total parameters when including the bias term in the depthwise convolution:
#    - Depthwise bias: 32 (for each input channel).
#    - Total parameters: 288 (depthwise weights) + 32 (depthwise biases) + 2048 (pointwise weights) + 64 (pointwise biases) = 2432.


# Q: Why not use depth-wise separable convolution here?
# A: RGB channels in input images are highly correlated, so regular convolution is preferred for capturing complex patterns between channels in the initial layers.


# GlobalAveragePooling2D reduces the spatial dimensions to a single value per feature map
x = layers.GlobalAveragePooling2D()(x)

# Dropout for regularization, reducing overfitting by randomly setting a fraction of input units to 0 during training
x = layers.Dropout(0.5)(x)

# Output layer with a sigmoid activation function, suitable for binary classification
outputs = layers.Dense(1, activation="sigmoid")(x)




    The first layer acts as a collection of various edge detectors.

    As you go deeper, the activations become increasingly abstract and less visually interpretable. They begin to encode higher-level concepts such as “cat ear” and “cat eye.”

    The sparsity of the activations increases with the depth of the layer: in the first layer, almost all filters are activated by the input image, but in the following layers, more and more filters are blank. This means the pattern encoded by the filter isn’t found in the input image


To calculate the trainable parameters in a depthwise separable convolution layer, we need to consider the two parts of the operation: depthwise convolution and pointwise convolution, including the biases.

Given:
Input size: 
7
×
7
×
3
7×7×3 (height 
×
× width 
×
× channels).
Filter size: 
3
×
3
3×3.
Number of filters: 2.
Bias: Each filter has one bias.
Depthwise Convolution:
Applies a single 
3
×
3
3×3 filter to each channel independently.
Total filters: 
1
1 filter per channel, so there are 
3
3 depthwise filters.
Trainable parameters for depthwise convolution:
3
×
3
 (filter size)
×
3
 (channels)
=
27
 weights
.
3×3 (filter size)×3 (channels)=27 weights.
Biases for depthwise convolution:
3
 (one bias per channel)
.
3 (one bias per channel).
Total trainable parameters for depthwise convolution: 
27
+
3
=
30
27+3=30.
Pointwise Convolution:
Applies 
1
×
1
1×1 convolutions to combine the outputs of the depthwise convolution across channels.
Output channels: 2 (as specified in the problem).
Trainable parameters for pointwise convolution:
1
×
1
 (filter size)
×
3
 (input channels)
×
2
 (output channels)
=
6
 weights
.
1×1 (filter size)×3 (input channels)×2 (output channels)=6 weights.
Biases for pointwise convolution:
2
 (one bias per output channel)
.
2 (one bias per output channel).
Total trainable parameters for pointwise convolution: 
6
+
2
=
8
6+2=8.
Total Trainable Parameters:
Depthwise parameters
+
Pointwise parameters
=
30
+
8
=
38.
Depthwise parameters+Pointwise parameters=30+8=38.
Final Answer:
The total count of trainable parameters in this depthwise separable convolution layer is 38.
