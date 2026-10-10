"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    token_to_id = {}
    
    for i ,special_token in enumerate(specials):
        token_to_id[special_token] = i

    j = len(specials)
    all_tokens = []
    for sentence_str in sentences:
        all_tokens.extend(sentence_str.split())
    for token in all_tokens:
        if token not in token_to_id:
            token_to_id[token] = j
            j+=1
    
    return token_to_id

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    id_to_token = {v:i for i , v in token_to_id.items()}

    return id_to_token

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    sentence_str = sentence.split()
    
    ids = [token_to_id.get(s, token_to_id[unk_token]) for s in sentence_str]


    return ids

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    tokens = [ id_to_token[i] for i in ids]

    return tokens

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    
    tokens = ids[:max_len]

    if len(tokens) < max_len:
        pad_tokens = tokens + [pad_id] * (max_len - len(tokens))
    

    return pad_tokens

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    
    batch = torch.tensor(padded_sequences,dtype = torch.long)

    return batch

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    scaled_embeddings = embeddings * math.sqrt(d_model)

    return scaled_embeddings

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    dt = []
    if d_model % 2 ==0:
        for i in range((d_model//2)):
            dt += torch.exp(2*i * -torch.log(torch.tensor([10000]))/d_model)

        return torch.tensor(dt)

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    p = torch.arange(max_len,dtype = torch.float)

    p_reshaped = p.unsqueeze(-1)

    return p_reshaped

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    if position.dim() == 1:
        position.unsqueeze(1)

    pe[:,::2] = torch.sin(position * div_term)

    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
   if position.dim() == 1:
        position.unsqueeze(1)

   pe[:,1::2] = torch.cos(position * div_term)

   return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    pe = torch.zeros((max_len,d_model))

    position = build_position_index_column(max_len)

    div_term = compute_positional_div_term(d_model)

    pe = fill_even_indices_with_sin(pe,position,div_term)
    pe = fill_odd_indices_with_cos(pe,position,div_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    L = embedded_batch.shape[1]

    out = embedded_batch + positional_encoding[:L,:]
    return out

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    
    B, L = token_ids.shape
    m =  torch.where(token_ids != pad_id, True, False)
    m_reshaped = m.reshape(B,1,1,L)

    return m_reshaped

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask_reshaped = mask.reshape(1,1,seq_len,seq_len)

    return mask_reshaped

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    
    raw_attn = torch.matmul(query,key.transpose(-2,-1))

    return raw_attn

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill(mask == False,float('-inf'))

# Step 20 - softmax_attention_weights
import torch
import torch.nn.functional as F

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    attn_weights = F.softmax(masked_scores , dim = -1)

    attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)

    return attn_weights

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    
    return torch.matmul(attention_weights, value)

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    scores = compute_raw_attention_scores(query,key)
    d_k = key.shape[-1]
    scaled_raw_attn = scale_attention_scores(scores,d_k)

    if mask is not None:
        scaled_raw_attn = mask_attention_scores_with_neg_inf(scaled_raw_attn,mask)
    

    attn_weights = softmax_attention_weights(scaled_raw_attn)
    ctx = apply_attention_weights_to_values(attn_weights,value)

    return ctx, attn_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    
    B , L , d_model = tensor.shape

    return tensor.reshape(B,L,num_heads,d_model//num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    
    return split_tensor.transpose(1,2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    
    multi_head_tensor_transpose = multi_head_tensor.transpose(1,2)

    B,L,H,d_k = multi_head_tensor_transpose.shape

    d_model = H*d_k

    return multi_head_tensor_transpose.reshape(B,L,d_model)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    
    out = x @ weight.transpose(-1,-2)

    return out if bias is None else out + bias

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    
    q = apply_linear_projection(x,w_q,b_q)
    k = apply_linear_projection(x,w_k,b_k)
    v = apply_linear_projection(x,w_v,b_v)

    return (q,k,v)

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    return tuple(
        transpose_heads_before_sequence(split_last_dim_into_heads(x, num_heads))
        for x in (q, k, v)
    )

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    
    return scaled_dot_product_attention(q_h,k_h,v_h,mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    
    return apply_linear_projection(merge_heads_back_to_model_dim(context),w_o,b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    
    # If query, key, and value are the same tensor (self-attention)
    if query is key and key is value:
        q, k, v = project_to_query_key_value(query, w_q, None, w_k, None, w_v, None)
    else:
        # Cross-attention: project query and key/value separately
        q, _, _ = project_to_query_key_value(query, w_q, None, w_k, None, w_v, None)
        _, k, v = project_to_query_key_value(key, w_q, None, w_k, None, w_v, None)

    q_h , k_h , v_h = split_qkv_into_heads(q,k,v,num_heads)

    ctx , attn = multi_head_scaled_dot_product_attention(q_h,k_h,v_h,mask)

    out = merge_heads_and_project_output(ctx,w_o,None)

    return out

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    
    return torch.relu(x @ w1 + b1)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).

    return apply_ffn_second_linear(apply_ffn_first_linear_and_relu(x,w1,b1),w2,b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    
    var,mean =  torch.var_mean(x,dim=-1,correction=0,keepdim=True)

    return mean, var

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    
    mean , var = compute_layer_norm_mean_and_variance(x)

    x_hat = (x-mean) / torch.sqrt(var+eps)

    y = gamma * x_hat + beta

    return y

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    
    return normalize_and_scale_with_gamma_beta((residual_input+sublayer_output),gamma,beta,eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    
    return x * keep_mask.to(x.dtype) * (1.0 / keep_prob)

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    sublayer_out = assemble_multi_head_attention_forward(x,x,x,w_q,w_k,w_v,w_o,num_heads,src_mask)

    return apply_residual_add_and_norm(x,sublayer_out,gamma,beta)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    out_sublayer = position_wise_feed_forward_network(x,w1,b1,w2,b2)

    return apply_residual_add_and_norm(x,out_sublayer,gamma,beta)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    # Separate parameters for attention sublayer
    attn_params = {
        'w_q': layer_params['w_q'],
        'w_k': layer_params['w_k'],
        'w_v': layer_params['w_v'],
        'w_o': layer_params['w_o'],
        'gamma': layer_params['attn_gamma'],
        'beta': layer_params['attn_beta'],
    }
    
    # Separate parameters for feed-forward sublayer
    ffn_params = {
        'w1': layer_params['w1'],
        'b1': layer_params['b1'],
        'w2': layer_params['w2'],
        'b2': layer_params['b2'],
        'gamma': layer_params['ffn_gamma'],
        'beta': layer_params['ffn_beta'],
    }

    h = encoder_layer_self_attention_sublayer(x, num_heads = num_heads, src_mask = src_mask,**attn_params)

    y = encoder_layer_feed_forward_sublayer(h,**ffn_params)

    return y

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    H = x 
    for layer_param in encoder_layer_params_list:
        H = assemble_encoder_layer(H,layer_param,num_heads,src_mask)

    return H

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    sublayer_out = assemble_multi_head_attention_forward(y,y,y,w_q,w_k,w_v,w_o,num_heads,tgt_mask)

    return apply_residual_add_and_norm(y,sublayer_out,gamma,beta)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    
    if src_mask is not None and src_mask.dim() == 2:
     src_mask = src_mask.unsqueeze(1).unsqueeze(2)


    out_sublayer = assemble_multi_head_attention_forward(y,encoder_output,encoder_output,w_q,w_k,w_v,w_o,num_heads,src_mask)

    return apply_residual_add_and_norm(y,out_sublayer,gamma,beta)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    out_sublayer = position_wise_feed_forward_network(y,w1,b1,w2,b2)

    return apply_residual_add_and_norm(y,out_sublayer,gamma,beta)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN.

    layer_params keys (all torch tensors):
      masked self-attention : w_q_self, w_k_self, w_v_self, w_o_self, self_gamma, self_beta
      cross-attention       : w_q_cross, w_k_cross, w_v_cross, w_o_cross, cross_gamma, cross_beta
      feed-forward          : w1, b1, w2, b2, ffn_gamma, ffn_beta
    """
    # TODO: chain the three decoder sublayers using params from layer_params.
    
    masked_attn_param = {
      'w_q' : layer_params['w_q_self'],
      'w_k' : layer_params['w_k_self'],
      'w_v' : layer_params['w_v_self'],
      'w_o' : layer_params['w_o_self'],
      'gamma': layer_params['self_gamma'],
      'beta': layer_params['self_beta'] 
    }

    corss_attn_param = {
      'w_q' : layer_params['w_q_cross'],
      'w_k' : layer_params['w_k_cross'],
      'w_v' : layer_params['w_v_cross'],
      'w_o' : layer_params['w_o_cross'],
      'gamma' : layer_params['cross_gamma'],
      'beta': layer_params['cross_beta']
    }

    ffn_param = {
      'w1' : layer_params['w1'],
      'b1' : layer_params['b1'],
      'w2' : layer_params['w2'],
      'b2' : layer_params['b2'],
      'gamma': layer_params['ffn_gamma'],
      'beta' : layer_params['ffn_beta']
    }



    self_out = decoder_layer_masked_self_attention_sublayer(y,num_heads = num_heads , tgt_mask = tgt_mask, **masked_attn_param)

    cross_out = decoder_layer_cross_attention_sublayer(self_out,encoder_output,num_heads = num_heads,src_mask = src_mask,**corss_attn_param)

    ffn_out = decoder_layer_feed_forward_sublayer(cross_out,**ffn_param)

    return ffn_out

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    
    H = y

    for layer_param in decoder_layer_params_list:
        H = assemble_decoder_layer(H,encoder_output,layer_param,num_heads,src_mask,tgt_mask)

    return H

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    

    return apply_linear_projection(decoder_output,output_projection_weight,output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.transpose(-1,-2)

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    
    return torch.log_softmax(logits,dim= -1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
   
   # --- STEP 1: EMBEDDING & POSITIONAL ENCODING ---

    # 1. Lookup and scale source embeddings
    src_embed = model_params['token_embedding'][src_ids]
    scaled_src_embed = scale_embeddings_by_sqrt_d_model(src_embed, src_embed.shape[-1])

    # 2. Lookup and scale target embeddings
    tgt_embed = model_params['token_embedding'][tgt_ids]
    scaled_tgt_embed = scale_embeddings_by_sqrt_d_model(tgt_embed, tgt_embed.shape[-1])

    # 3. Add positional encodings to scaled embeddings
    src_pe = build_sinusoidal_positional_encoding(src_embed.shape[1], src_embed.shape[-1])
    src_out = add_positional_encoding_to_embeddings(scaled_src_embed, src_pe)

    tgt_pe = build_sinusoidal_positional_encoding(tgt_embed.shape[1], tgt_embed.shape[-1])
    tgt_out = add_positional_encoding_to_embeddings(scaled_tgt_embed, tgt_pe)
    
    # --- STEP 2: MASK GENERATION ---

    # Build source padding mask
    src_mask = build_padding_mask(src_ids,pad_id)
    # Build target padding mask and lower-triangular causal mask
    tgt_pad = build_padding_mask(tgt_ids,pad_id)
    tgt_causal = build_causal_mask(tgt_embed.shape[1])
    # Combine target padding and causal masks
    tgt_mask = combine_padding_and_causal_masks(tgt_pad,tgt_causal)
    
    # --- STEP 3: ENCODER & DECODER STACKS ---

    # Process source sequence through the N encoder layers
    encoder_out = stack_encoder_layers(src_out,model_params['encoder_layers'],num_heads,src_mask)
    # Process target sequence through the N decoder layers
    decoder_out = stack_decoder_layers(tgt_out,encoder_out,model_params['decoder_layers'],num_heads,src_mask,tgt_mask)
    
    # --- STEP 4: OUTPUT PROJECTION & LOG-SOFTMAX ---

    # Project decoder states to vocabulary logits
    logits = apply_final_output_projection(decoder_out,model_params['output_projection'])
    # Compute log probabilities acros
    log_probs = apply_log_softmax_over_vocab(logits)

    return log_probs

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    std = 1.0 / math.sqrt(d_model)
    std_ff = 1.0 / math.sqrt(d_ff)

    return {
        "w_q": (torch.randn(d_model, d_model) * std).detach().requires_grad_(True),
        "w_k": (torch.randn(d_model, d_model) * std).detach().requires_grad_(True),
        "w_v": (torch.randn(d_model, d_model) * std).detach().requires_grad_(True),
        "w_o": (torch.randn(d_model, d_model) * std).detach().requires_grad_(True),
        "w1": (torch.randn(d_model, d_ff) * std).detach().requires_grad_(True),
        "b1": torch.zeros(d_ff, requires_grad=True),
        "w2": (torch.randn(d_ff, d_model) * std_ff).detach().requires_grad_(True),
        "b2": torch.zeros(d_model, requires_grad=True),
        "attn_gamma": torch.ones(d_model, requires_grad=True),
        "attn_beta": torch.zeros(d_model, requires_grad=True),
        "ffn_gamma": torch.ones(d_model, requires_grad=True),
        "ffn_beta": torch.zeros(d_model, requires_grad=True),
    }

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    return {
        # Masked Self-Attention projections
        "w_q_self": torch.randn(d_model, d_model, requires_grad=True),
        "w_k_self": torch.randn(d_model, d_model, requires_grad=True),
        "w_v_self": torch.randn(d_model, d_model, requires_grad=True),
        "w_o_self": torch.randn(d_model, d_model, requires_grad=True),

        # Cross-Attention projections
        "w_q_cross": torch.randn(d_model, d_model, requires_grad=True),
        "w_k_cross": torch.randn(d_model, d_model, requires_grad=True),
        "w_v_cross": torch.randn(d_model, d_model, requires_grad=True),
        "w_o_cross": torch.randn(d_model, d_model, requires_grad=True),

        # Feed-Forward Network
        "w1": torch.randn(d_model, d_ff, requires_grad=True),
        "b1": torch.zeros(d_ff, requires_grad=True),
        "w2": torch.randn(d_ff, d_model, requires_grad=True),
        "b2": torch.zeros(d_model, requires_grad=True),

        # LayerNorm parameters (gamma=1, beta=0)
        "self_gamma": torch.ones(d_model, requires_grad=True),
        "self_beta": torch.zeros(d_model, requires_grad=True),
        "cross_gamma": torch.ones(d_model, requires_grad=True),
        "cross_beta": torch.zeros(d_model, requires_grad=True),
        "ffn_gamma": torch.ones(d_model, requires_grad=True),
        "ffn_beta": torch.zeros(d_model, requires_grad=True),
    }

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    src_embed = torch.randn(vocab_size, d_model, requires_grad=True)
    tgt_embed = torch.randn(vocab_size, d_model, requires_grad=True)
    
    if tie_weights:
        output_projection = tgt_embed
    else:
        output_projection = torch.randn(vocab_size, d_model, requires_grad=True)
    
    return {
        'src_embedding': src_embed,
        'tgt_embedding': tgt_embed,
        'output_projection': output_projection
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(
    encoder_layer_params,
    decoder_layer_params,
    embedding_params
):
    params = []
    seen = set()

    def add_parameters(param_dict):
        for tensor in param_dict.values():
            if id(tensor) not in seen:
                seen.add(id(tensor))
                params.append(tensor)

    # Preserve required order
    for layer_params in encoder_layer_params:
        add_parameters(layer_params)

    for layer_params in decoder_layer_params:
        add_parameters(layer_params)

    add_parameters(embedding_params)

    return params

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    
    batch_size = target_ids.size(0)

    start_tokens = torch.tensor([[start_token_id]]).expand(batch_size, -1)

    return torch.cat((start_tokens, target_ids[:, :-1]), dim=1)

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    
    lr = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return lr

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    val = epsilon / (vocab_size-2)

    return torch.full(shape, val , dtype = torch.float32)

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    
    # 1. Clone to avoid modifying input in-place
    smoothed_dist = smoothed_distribution.clone()
    
    # 2. Reshape indices from (B, T) to (B, T, 1)
    gold_indices = gold_token_ids.unsqueeze(-1)
    
    # 3. Create a tensor matching gold_indices shape filled with confidence value
    src = torch.full_like(gold_indices, confidence, dtype=smoothed_dist.dtype)
    
    # 4. Scatter confidence into the vocabulary dimension (dim=-1)
    smoothed_dist.scatter_(dim=-1, index=gold_indices, src=src)
    
    return smoothed_dist

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
    
    smoothed_dis = smoothed_distribution.clone()

    # column mask
    smoothed_dis[:,:,pad_id] = 0.0

    row_mask = (gold_token_ids.to(torch.long) == pad_id)

    smoothed_dis[row_mask] = 0.0

    return smoothed_dis

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    
    return torch.sum(-log_probabilities*smoothed_distribution)

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids

    # count real tokens
    N = gold_token_ids[(gold_token_ids.to(torch.long) != pad_id)].shape[0]
    
    return total_loss / max(N,1)

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only
    
    pred = torch.argmax(log_probabilities,dim=-1)
    mask = (gold_token_ids != pad_id)
    correct = (pred == gold_token_ids) & mask
    return correct.sum() / max(mask.sum(),1)

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
   
    m = [torch.zeros_like(param) for param in parameter_list]
    v = [torch.zeros_like(param) for param in parameter_list]
    t = 0

    return {
        "m":m,
        "v":v,
        "t":t
    }

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    
    m_t = beta1 * m_prev + (1-beta1) * grad
    
    return m_t

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    v_t = beta2 * v_prev + (1 - beta2) * grad ** 2

    return v_t

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    
    m_hat = m_t / (1-beta1**step)
    v_hat = v_t / (1-beta2**step)

    return m_hat,v_hat

# Step 68 - compute_adam_parameter_update
import torch

def compute_adam_parameter_update(m_hat, v_hat, learning_rate, epsilon):
    """Return delta = learning_rate * m_hat / (sqrt(v_hat) + epsilon); the caller subtracts it."""
    # TODO: compute the Adam step from the bias-corrected moments without tracking gradients
    
    delta = learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

    return delta

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    # TODO: increment t, then for each param with a grad update m, v, bias-correct, and subtract delta in place.
    
    # 1. Increment step counter
    optimizer_state["t"] += 1
    t = optimizer_state["t"]

    with torch.no_grad():
        # 2. Zip params with their corresponding m and v buffers
        for i, (param, m, v) in enumerate(
            zip(parameter_list, optimizer_state["m"], optimizer_state["v"])
        ):
            # Skip unused parameters with no gradient
            if param.grad is None:
                continue

            # 3. Update raw first and second moments
            m_new = update_adam_first_moment(m, param.grad, beta1)
            v_new = update_adam_second_moment(v, param.grad, beta2)

            # Persist updated raw buffers back to state
            optimizer_state["m"][i] = m_new
            optimizer_state["v"][i] = v_new

            # 4. Compute bias-corrected moments
            m_hat, v_hat = apply_adam_bias_correction(
                m_new, v_new, beta1, beta2, t
            )

            # 5. Compute per-parameter delta
            delta = compute_adam_parameter_update(
                m_hat, v_hat, learning_rate, epsilon
            )

            # 6. Apply delta in-place to the model parameter
            param.sub_(delta)

    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    # TODO: clear the accumulated gradient on every parameter tensor in the list
    for param in parameter_list:
        if param.grad is not None:
            param.grad = None

# Step 71 - compute_batch_training_loss
def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    # Shift targets right for teacher forcing
    decoder_input = shift_targets_right_with_start_token(
        tgt_batch,
        config["start_id"],
    )

    # Full Transformer forward pass
    logits = run_transformer_forward(
        src_batch,
        decoder_input,
        model_params,
        config["num_heads"],
        config['pad_id']
    )

    # Create smoothed target distribution with the same shape as logits
    target_distribution = build_uniform_smoothing_distribution(
        logits.shape,
        config["vocab_size"],
        config["smoothing"],
    )

    # Give the gold token its confidence
    target_distribution = set_confidence_on_gold_tokens(
        target_distribution,
        tgt_batch,
        1.0 - config["smoothing"],
    )

    # Remove padding from the distribution
    target_distribution = zero_pad_column_and_pad_token_rows(
        target_distribution,
        tgt_batch,
        config["pad_id"],
    )

    # KL loss for every token
    token_losses = compute_label_smoothed_kl_loss(
        logits,
        target_distribution,
    )

    # Average only over non-padding tokens
    return average_loss_over_non_pad_tokens(
        token_losses,
        tgt_batch,
        config["pad_id"],
    )

# Step 72 - run_training_step_with_backprop
def run_training_step_with_backprop(
    src_batch,
    tgt_batch,
    parameter_list,
    model_params,
    optimizer_state,
    step_number,
    config,
):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.

    Returns the scalar loss value for the step as a Python float.
    """

    # 1. Clear gradients from the previous iteration
    zero_all_parameter_gradients(parameter_list)

    # 2. Forward pass
    loss = compute_batch_training_loss(
        src_batch,
        tgt_batch,
        model_params,
        config,
    )

    # 3. Backpropagation
    loss.backward()

    # 4. Compute Noam learning rate for this step
    lr = compute_noam_learning_rate(
        step_number,
        config["d_model"],
        config["warmup_steps"],
    )

    # 5. Adam update across ALL parameters
    optimizer_state = apply_adam_step_to_all_parameters(
        parameter_list,
        optimizer_state,
        lr,
        beta1 = 0.9,
        beta2 = 0.98,
        epsilon = 1e-9
    )

    # 6. Return a Python float for logging
    return loss.item()

# Step 73 - run_training_loop_for_steps
def run_training_loop_for_steps(
    batches,
    parameter_list,
    model_params,
    optimizer_state,
    num_steps,
    config,
):
    """Run num_steps training iterations, cycling through batches, and return per-step losses."""

    losses = []

    for i in range(num_steps):
        # Cycle through batches: 0, 1, 2, ..., 0, 1, ...
        batch_index = i % len(batches)

        src_batch, tgt_batch = batches[batch_index]

        # Training steps must start at 1, not 0
        step_number = i + 1

        loss = run_training_step_with_backprop(
            src_batch,
            tgt_batch,
            parameter_list,
            model_params,
            optimizer_state,
            step_number,
            config,
        )

        losses.append(loss)

    return losses

# Step 74 - pick_next_token_by_argmax
import torch

def pick_next_token_by_argmax(final_step_logits):
    """Greedy: return argmax token id per batch row.

    final_step_logits: FloatTensor of shape (batch, vocab_size)
    returns: LongTensor of shape (batch,)
    """
    # TODO: pick the next greedy token id by taking the argmax over the vocab axis
    
    return final_step_logits.argmax(-1)

# Step 75 - compute_length_penalty
def compute_length_penalty(sequence_length, alpha):
    # TODO: return the Google NMT length penalty for the given sequence_length and alpha.
    
    lp = ((5+sequence_length)/6)**alpha

    return lp

# Step 76 - compute_candidate_scores
import torch

def compute_candidate_scores(beam_scores, next_token_log_probs):
    # TODO: add each beam's running log-prob to its row of next-token log probs.
    
    return beam_scores.unsqueeze(-1) + next_token_log_probs

# Step 77 - select_top_k_candidates
import torch

def select_top_k_candidates(candidate_scores, k):
    # TODO: pick the top k (beam_index, token_id, score) triples from candidate_scores
    v = candidate_scores.shape[-1]
    candidate_scores = candidate_scores.flatten()
    scores,indices = candidate_scores.topk(k)

    beam_indices = indices // v
    token_ids = indices % v
    return {
        "beam_indices":beam_indices,
        "token_ids":token_ids,
        "scores":scores
        }

# Step 78 - append_tokens_to_beam_sequences
import torch

def append_tokens_to_beam_sequences(beam_sequences, beam_indices, token_ids):
    # TODO: gather parent beam rows and append the new token ids as the last column
    
    beam_sequences = torch.concat((beam_sequences[beam_indices]  , token_ids.unsqueeze(1) ),  dim=1 )

    return beam_sequences

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

