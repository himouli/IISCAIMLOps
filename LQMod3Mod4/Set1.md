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

Medium Complexity Questions:

How does layer normalization differ from batch normalization?
Answer: Layer normalization normalizes across features within a single example, while batch normalization normalizes across the batch dimension.

What is the primary limitation of recurrent neural networks (RNNs)?
Answer: They struggle with long-range dependencies due to vanishing gradients.

Why is self-attention more efficient than RNNs for long-range dependencies?
Answer: Self-attention processes all words in parallel, avoiding sequential dependencies.

How does BERT differ from traditional transformers?
Answer: BERT is bidirectional, meaning it attends to both left and right context, unlike autoregressive transformers.

Explain the significance of dropout in transformer architectures.
Answer: Dropout prevents overfitting by randomly deactivating a subset of neurons during training.

How does positional encoding work in Transformers?
Answer: It adds sine and cosine functions to word embeddings to encode position information.

Why are larger models like GPT-3 prone to hallucination?
Answer: They rely on pattern matching rather than explicit reasoning, leading to confident but incorrect outputs.

What is tokenization, and why is it crucial in NLP?
Answer: Tokenization splits text into meaningful units (tokens) for processing.

How does fine-tuning differ from training from scratch?
Answer: Fine-tuning adapts a pre-trained model to a specific task, reducing training time and required data.

