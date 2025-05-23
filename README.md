# PyTorch Transformer Model

This repository contains a basic implementation of the Transformer model architecture as described in the paper **"Attention Is All You Need"** by Vaswani et al. (2017), built using PyTorch.

## 📦 Components Implemented

- **Multi-Head Attention**: Jointly attends to information from different representation subspaces.
- **Positional Encoding**: Injects positional information into the token embeddings.
- **Position-wise Feed-Forward Networks (FFN)**: Two linear layers with ReLU, applied to each position independently.
- **Add & Norm**: Residual connections followed by layer normalization.
- **Encoder Layer**: Contains self-attention and FFN blocks with Add & Norm.
- **Encoder**: Stack of multiple encoder layers.
- **Decoder Layer**: Contains masked self-attention, encoder-decoder attention, and FFN blocks.
- **Decoder**: Stack of multiple decoder layers.
- **Transformer**: Combines encoder and decoder, includes input embeddings and final linear layer.

## 🧱 Code Structure

Main classes included:
- `MultiHeadAttention`
- `PositionalEncoding`
- `PositionwiseFeedForward`
- `AddNorm`
- `EncoderLayer`
- `Encoder`
- `DecoderLayer`
- `Decoder`
- `Transformer` (main model class)

Helper functions:
- `create_padding_mask`
- `create_look_ahead_mask`

## 🧰 Dependencies

- PyTorch

Install PyTorch: [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

## 🚀 Usage

1. Instantiate the `Transformer` class with parameters like number of layers, model dimensions, etc.
2. Prepare your training data (source and target sequences).
3. Create attention masks using provided helper functions.
4. Perform forward pass and use the output logits for training.

### Example
```python
# from your_module import Transformer, create_padding_mask, create_look_ahead_mask

model = Transformer(
    num_layers=6,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    input_vocab_size=src_vocab_size,
    target_vocab_size=tgt_vocab_size,
    max_seq_len=5000,
    dropout_rate=0.1
)

# Assuming dummy_src, dummy_tgt, and PAD_IDX are defined
src_padding_mask = create_padding_mask(dummy_src, PAD_IDX)
tgt_padding_mask = create_padding_mask(dummy_tgt, PAD_IDX)
look_ahead_mask = create_look_ahead_mask(dummy_tgt.size(1))
tgt_mask = tgt_padding_mask * (look_ahead_mask == 0)

logits, attention_weights = model(dummy_src, dummy_tgt, src_padding_mask, tgt_mask)
