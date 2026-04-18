"""
Golden dataset: 20 hand-written Q&A pairs from 'Attention Is All You Need'.
Creates a Langfuse dataset and populates it once.
Run this script directly: python -m agentic_rag.evaluation.dataset
"""

import logging
from agentic_rag.observability.langfuse_client import get_langfuse

logger = logging.getLogger(__name__)

DATASET_NAME = "attention_paper_golden_v1"

# 20 golden Q&A pairs grounded in the Attention Is All You Need paper.
# reference = the ground-truth answer your RAG system should be close to.
GOLDEN_PAIRS = [
    {
        "question": "What is the core idea behind the Transformer architecture?",
        "reference": "The Transformer relies entirely on attention mechanisms to draw global dependencies between input and output, dispensing with recurrence and convolutions entirely.",
    },
    {
        "question": "Why did the authors remove recurrence from the Transformer?",
        "reference": "Recurrence processes tokens sequentially, which prevents parallelization during training. Removing it allows all positions to be processed simultaneously, reducing training time significantly.",
    },
    {
        "question": "What is scaled dot-product attention?",
        "reference": "Scaled dot-product attention computes attention weights by taking the dot product of queries and keys, dividing by the square root of the key dimension to prevent vanishing gradients, then applying softmax and multiplying by the values.",
    },
    {
        "question": "Why is the dot product divided by the square root of dk in attention?",
        "reference": "For large values of dk, the dot products grow large in magnitude, pushing the softmax into regions with very small gradients. Dividing by sqrt(dk) counteracts this effect.",
    },
    {
        "question": "What is multi-head attention and why is it used?",
        "reference": "Multi-head attention runs scaled dot-product attention in parallel across h different learned projection subspaces, then concatenates the outputs. This allows the model to jointly attend to information from different representation subspaces at different positions.",
    },
    # {
    #     "question": "How many attention heads did the authors use in the base Transformer model?",
    #     "reference": "The base model uses 8 parallel attention heads, with dk and dv both equal to 64.",
    # },
    # {
    #     "question": "What are the three types of attention used in the Transformer?",
    #     "reference": "The Transformer uses encoder self-attention (each encoder position attends to all encoder positions), decoder self-attention (masked to prevent attending to future positions), and encoder-decoder attention (decoder attends to all encoder positions).",
    # },
    # {
    #     "question": "What is the purpose of masking in the decoder's self-attention?",
    #     "reference": "Masking prevents positions in the decoder from attending to subsequent positions, ensuring the auto-regressive property: predictions for position i can only depend on outputs at positions less than i.",
    # },
    # {
    #     "question": "What is positional encoding and why is it needed in the Transformer?",
    #     "reference": "Since the Transformer contains no recurrence or convolution, it has no inherent notion of token order. Positional encodings are added to input embeddings to inject information about the relative or absolute position of tokens in the sequence.",
    # },
    # {
    #     "question": "What functions were used for positional encoding in the paper?",
    #     "reference": "The authors used sine and cosine functions of different frequencies: PE(pos, 2i) = sin(pos/10000^(2i/dmodel)) and PE(pos, 2i+1) = cos(pos/10000^(2i/dmodel)).",
    # },
    # {
    #     "question": "What is the role of the feed-forward network in each Transformer layer?",
    #     "reference": "Each encoder and decoder layer contains a position-wise fully connected feed-forward network applied identically and independently to each position, consisting of two linear transformations with a ReLU activation in between.",
    # },
    # {
    #     "question": "What is the dimensionality of the feed-forward inner layer in the base model?",
    #     "reference": "The inner layer of the feed-forward network has dimensionality dff = 2048, while the input and output dimensionality is dmodel = 512.",
    # },
    # {
    #     "question": "How does the Transformer encoder differ from the decoder structurally?",
    #     "reference": "The encoder is composed of 6 identical layers each with two sub-layers: multi-head self-attention and a feed-forward network. The decoder has the same but adds a third sub-layer that performs multi-head attention over the encoder output, and uses masked self-attention.",
    # },
    # {
    #     "question": "What regularization techniques did the authors use during training?",
    #     "reference": "The authors applied residual dropout to the output of each sub-layer before it is added to the sub-layer input and normalized, and also applied dropout to the sums of the embeddings and positional encodings. A label smoothing value of 0.1 was also used.",
    # },
    # {
    #     "question": "What optimizer and learning rate schedule was used to train the Transformer?",
    #     "reference": "The Adam optimizer was used with beta1=0.9, beta2=0.98, and epsilon=1e-9. The learning rate increased linearly for warmup_steps then decreased proportionally to the inverse square root of the step number.",
    # },
    # {
    #     "question": "What BLEU score did the base Transformer achieve on WMT 2014 English-to-German translation?",
    #     "reference": "The base Transformer model achieved 27.3 BLEU on the WMT 2014 English-to-German translation task.",
    # },
    # {
    #     "question": "What BLEU score did the big Transformer achieve on WMT 2014 English-to-German translation?",
    #     "reference": "The big Transformer model achieved 28.4 BLEU on the WMT 2014 English-to-German translation task, outperforming all previously reported models.",
    # },
    # {
    #     "question": "How does the computational complexity of self-attention compare to recurrent layers?",
    #     "reference": "Self-attention has O(n^2 * d) complexity per layer versus O(n * d^2) for recurrent layers, where n is sequence length and d is representation dimensionality. Self-attention is faster when n is smaller than d, which is common in sentence-level tasks.",
    # },
    # {
    #     "question": "What is residual connection and layer normalization, and how are they applied in the Transformer?",
    #     "reference": "Each sub-layer output is added to its input (residual connection) and then normalized using layer normalization: LayerNorm(x + Sublayer(x)). This is applied around each of the two sub-layers in the encoder and each of the three in the decoder.",
    # },
    # {
    #     "question": "What tasks beyond machine translation did the authors test the Transformer on?",
    #     "reference": "The authors tested the Transformer on English constituency parsing and found it generalized well, achieving competitive results even with limited task-specific tuning.",
    # },
]


def create_langfuse_dataset() -> None:
    """
    Creates the Langfuse dataset and uploads all 20 golden items.
    Safe to re-run: Langfuse deduplicates by dataset name.
    """
    langfuse = get_langfuse()

    # Create dataset (idempotent — no error if already exists)
    langfuse.create_dataset(
        name=DATASET_NAME,
        description="20 golden Q&A pairs from 'Attention Is All You Need' for offline eval.",
    )
    logger.info(f"Dataset '{DATASET_NAME}' ensured in Langfuse.")

    for i, pair in enumerate(GOLDEN_PAIRS):
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={"question": pair["question"]},
            expected_output={"reference": pair["reference"]},
        )
    
    langfuse.flush()
    logger.info(f"Uploaded {len(GOLDEN_PAIRS)} items to dataset '{DATASET_NAME}'.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_langfuse_dataset()
    print(f"Done. Dataset '{DATASET_NAME}' is live in Langfuse.")