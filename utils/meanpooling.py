from einops import reduce, repeat
from torch import Tensor, sum, clamp


def mean_pooling(
        model_output: Tensor,
        attention_mask: Tensor):
    """
    Mean pooling with attention mask

    Args:
    model_output: Sequence of hidden-states at the output of the last layer of the model.
    attention_mask: Mask to avoid performing attention on padding token indices. Negative values are treated as padding.

    Returns:
    Mean-pooled output
    """
    # Extracting token embeddings (assuming model_output[0] has shape [batch, seq_len, feature])
    token_embeddings = model_output[0]
    
    # Expand attention_mask to match the dimensions of token_embeddings
    input_mask_expanded = repeat(
        attention_mask,
        'b l -> b l f',
        f=token_embeddings.shape[-1])

    # Perform masked summation across the sequence length dimension,
    # and normalize by the sum of the mask
    sum_masked_embeddings = reduce(token_embeddings * input_mask_expanded,
                                   'b l f -> b f',
                                   'sum')
    mask_sum = reduce(input_mask_expanded,
                      'b l f -> b f',
                      'sum')
    output = sum_masked_embeddings / clamp(mask_sum, min=1e-9)
    return output
