Great question! The causal mask in the decoder is what prevents the decoder from seeing future words during training. It ensures that the model only attends to past and current words, but not future words.

What is a Causal Mask?
A causal mask (also called a triangular mask) is a matrix that blocks attention to future tokens in the decoder's self-attention mechanism.

This mask makes sure that, at position 
𝑡
t, the decoder can only attend to positions ≤ t but not any future words.

Why is it Needed?
When generating text, we predict one word at a time, so the decoder should not know upcoming words in advance. The causal mask forces the model to generate outputs in an autoregressive manner, meaning it learns to predict the next token only from past tokens.

Example of a Causal Mask
Let's assume we are translating "The cat sat" → "Le chat s'est" in French.
During training, the decoder receives the correct output shifted by one position (so it predicts each word given past words).

Now, let’s visualize a causal mask for a 4-word sentence:

Mask Matrix Example for 4 Tokens
Copy
Edit
1  0  0  0
1  1  0  0
1  1  1  0
1  1  1  1
1 → Allowed attention (can see itself and previous words)
0 → Blocked attention (cannot see future words)
What This Means
Word Position	Can Attend To	Cannot See
1st word	Itself	2nd, 3rd, 4th
2nd word	1st, 2nd	3rd, 4th
3rd word	1st, 2nd, 3rd	4th
4th word	1st, 2nd, 3rd, 4th	(Nothing is masked at the end)
How Does the Causal Mask Work in the Decoder?
During Training

The decoder receives the full target sentence (shifted right).
But each word can only see past words, thanks to the causal mask.
This prevents it from using future words to predict the current word.
During Inference (Generation)

The decoder generates one word at a time.
The mask ensures that the next word is predicted only based on previously generated words.
How the Causal Mask is Applied in Self-Attention
In self-attention, we compute a score matrix 
𝑄
𝐾
𝑇
QK 
T
 , which determines how much attention each word gives to others.

Before softmax, we apply the causal mask:
Future positions (upper triangle of the matrix) are set to negative infinity (-∞).
Softmax turns those -∞ values into zero probability, ensuring the decoder ignores future words.
Key Takeaways
✅ A causal mask prevents the decoder from seeing future words during training.
✅ It ensures autoregressive generation, making the model learn to predict text step by step.
✅ It blocks future words using a triangular matrix, allowing attention only to previous words.
✅ Without it, the decoder would "cheat" by looking ahead at future words.
