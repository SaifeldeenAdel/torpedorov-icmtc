# Team Torpedo ROV - ASR Documentation

## Data Pipeline

This is a description of how our data is read and loaded for inference and for training.

-   Reading and decoding audio files into tensors

-   Extracing spectrogram features using short fourier transform

-   Carrying out normalization and padding to ensure consistent data and no unexpected lengths and frequencies

-   Transcripts are split into characters and tokenized based on a list of arabic characters.

-   A beginning of sentence token "<" and end of sentence token ">" are added to the tokenized transcripts to be markers during training

-   Finally, tensorflow Datasets were created by combining processed audio (features) and processed transcripts where batches are then fed into our model.

## Model Architecture

We decided to employ a Transformer Encoder-Decoder based architecture

-   Token Embedding Layer: Combines token and position embeddings, adding these embeddings to the input sequences to capture positional information.

-   Speech Feature Embedding Layer: Applies three convolutional layers to process local relationships within audio features.

-   Transformer Encoder Layer: We utilize a multi-head attention mechanism to capture dependencies within input sequences of tokens, and incorporates a feed-forward neural network with ReLU activation, along with layer normalization and dropout for regularization.

-   Transformer Decoder Layer: Uses self-attention to ensure each token only sees previous tokens, applies attention mechanisms to both the decoder's own inputs and the encoder's outputs, and includes a neural network and normalization to process inputs.

-   Transformer Model: We integrate both the encoder and decoder layers to form the whole model that processes audio spectrograms as inputs and predicts character sequences, combining the outputs of the encoder and decoder to generate final predictions.

## Methodlogies

-   Greedy decoding strategy to generate predictions, by selecting the highest probability token from the logits and stopping decoding at the end token.

-   The use of Categorical Crossentropy with label smoothing as the loss function during training.

-   Applying one-hot encoding to target sequences for comparing with predicitions.

## To run inference

-   Download all requirements from `requirements.txt`

-   Ensure that the `/test` folder exists in the same directory as `inference.py` and `modelArch.py` and that it contains all the `.wav` files

-   Run `inference.py`, there are progress print statements showing which sample is being processed. It takes a while to run on all samples (~4 hours)

-   A `submission.csv` is produced with `audio` and `transcript` as column headers
