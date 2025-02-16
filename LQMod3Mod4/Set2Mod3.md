Long Quiz: Natural Language Processing (NLP)

Part A [10 x 3 points = 30] : Multiple Choice Questions (MCQs)

Which of the following is NOT a task in NLP?
A. Named Entity Recognition (NER)
B. Sentiment Analysis
C. Image Classification
D. Part-of-Speech Tagging

Answer: C

In the NLP pipeline, which step is responsible for removing stop words?
A. Tokenization
B. Lemmatization
C. Stopword Removal
D. Text Vectorization

Answer: C

Which library is commonly used for NLP tasks in Python?
A. OpenCV
B. NumPy
C. NLTK
D. TensorFlow

Answer: C

Which text representation method preserves the semantic meaning of words?
A. One-hot encoding
B. Bag of Words
C. Word Embeddings
D. Character-level tokenization

Answer: C

Byte Pair Encoding (BPE) is primarily used for:
A. Removing stop words
B. Tokenizing words into subwords
C. Lemmatization
D. Syntactic parsing

Answer: B

What is the main advantage of training word embeddings?
A. It reduces model accuracy
B. It allows words with similar meanings to have similar representations
C. It requires less training data
D. It eliminates the need for tokenization

Answer: B

Sequence modeling is used when:
A. The order of data does not matter
B. The order of data is important
C. The dataset is static
D. Features are independent

Answer: B

Which of the following is NOT a recurrent architecture?
A. RNN
B. LSTM
C. GRU
D. CNN

Answer: D

In the Attention Mechanism, which vector determines what information to focus on?
A. Query
B. Key
C. Value
D. Embedding

Answer: A

What is the role of Positional Embedding in Transformers?
A. To introduce word order information
B. To improve tokenization
C. To replace attention mechanisms
D. To filter stop words

Answer: A

Part B [4 x 2 points = 8] : Multiple Choice Questions (MCQs)

Which technique is commonly used to normalize activations in deep learning models?
A. Layer Normalization
B. Dropout
C. Word Embeddings
D. One-hot Encoding

Answer: A

What is the key advantage of Multi-Head Attention?
A. Reduces model complexity
B. Enables the model to focus on multiple aspects of the input simultaneously
C. Eliminates the need for positional encoding
D. Reduces memory requirements

Answer: B

What does BERT use for training?
A. Supervised learning only
B. Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
C. Unsupervised clustering
D. Word2Vec

Answer: B

Which of the following is a common application of BERT?
A. Image classification
B. Audio signal processing
C. Question answering
D. Reinforcement learning

Answer: C

Part C [2 x 6 points = 12]: MCQs with Explanation

Why is masked attention used in the Transformer decoder?
A. To allow the decoder to see future tokens during training
B. To prevent the decoder from attending to future tokens
C. To enhance computational efficiency
D. To improve embedding quality

Answer: B
Explanation: Masked attention ensures that the decoder predicts the next token based only on previously generated tokens, preventing information leakage.

What is the significance of Transformer Decoders in Neural Machine Translation (NMT)?
A. They directly translate input without processing
B. They generate text by attending to the encoded input while generating words step by step
C. They use convolutional layers instead of attention
D. They remove the need for tokenization

Answer: B
Explanation: Transformer Decoders process the encoded input and generate output tokens sequentially, using self-attention and masked attention mechanisms.

Medium to Hard Complexity Questions:

What is the impact of increasing the number of attention heads in a Transformer model?
Answer: It allows the model to capture different contextual relationships but increases computational complexity.

How does backpropagation work in attention-based architectures?
Answer: Gradients flow through attention weight computations, updating Query, Key, and Value matrices.

Why do transformers use layer normalization instead of batch normalization?
Answer: Layer normalization is independent of batch size and works better for sequential data.

What problem does gradient clipping solve in deep learning models?
Answer: It prevents exploding gradients by capping their values during backpropagation.

Explain the difference between self-attention and cross-attention.
Answer: Self-attention attends to different parts of the same sequence, while cross-attention attends to an external sequence (e.g., encoder-decoder interactions).

How does zero-shot learning work in NLP models like GPT?
Answer: The model generalizes to new tasks without explicit training by leveraging learned patterns from large datasets.

Why does fine-tuning large-scale models require lower learning rates?
Answer: To avoid catastrophic forgetting and preserve pre-trained knowledge.

What is the role of the feed-forward network in a Transformer block?
Answer: It applies non-linearity and learns additional features beyond attention-based representations.

How do mixture of experts (MoE) models enhance Transformer efficiency?
Answer: They use multiple expert networks, activating only a subset per input, reducing overall computation.

