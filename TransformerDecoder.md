Why is Masking Applied to the Self-Attention Layer in a Transformer Decoder?
In a Transformer decoder, masking is essential to ensure that the model does not "cheat" by looking at future words while generating output. This is especially important in auto-regressive tasks like text generation, where the model generates one word at a time.

🔹 The Problem: Why Do We Need Masking?
In self-attention, each token attends to all other tokens in the sequence.
This works fine in the encoder, but in the decoder, the model should only use past tokens (not future ones) when generating a word.
Without masking, the decoder could see future words, leading to data leakage (cheating) during training.
🔹 Solution: Masking in Self-Attention
To prevent the model from looking at future words, we apply a causal mask (look-ahead mask) that blocks attention to future tokens.

Example: Predicting "The cat sat on the mat"
Let's assume the decoder is at different time steps.

Time Step	Visible Tokens (After Masking)	Hidden Tokens
1	"The"	"cat sat on the mat"
2	"The cat"	"sat on the mat"
3	"The cat sat"	"on the mat"
💡 The model can only attend to words on its left (previous words) and NOT on the right (future words).

🔹 How Masking Works in Practice
The mask is a triangular matrix that blocks future positions by setting them to negative infinity (-∞) in the attention score matrix.
This ensures that the softmax function assigns near-zero attention to masked positions.
Masked Attention Matrix (Example for a 4-word sequence)
[
1
0
0
0
1
1
0
0
1
1
1
0
1
1
1
1
]
​
  
1
1
1
1
​
  
0
1
1
1
​
  
0
0
1
1
​
  
0
0
0
1
​
  
​
 
1 → Allowed (attend to previous words)
0 → Masked (future words are hidden)
🔹 Types of Masks in Transformers
Type	Used In	Purpose
Padding Mask	Encoder & Decoder	Ignores PAD tokens (useful for variable-length sequences)
Causal Mask (Look-Ahead Mask)	Decoder Only	Blocks future words in self-attention
🔹 Why Is This Important?
✅ Ensures sequential token prediction → The model only sees past words while predicting the next word.
✅ Prevents information leakage → Helps the decoder learn in a realistic auto-regressive manner.
✅ Enables accurate text generation → Used in GPT models, chatbots, and machine translation.

🔹 Summary
Masking prevents the decoder from attending to future words in self-attention.
It ensures that the model predicts words sequentially, just like humans write text.
Without masking, the decoder could see the entire sequence, making training unrealistic.
