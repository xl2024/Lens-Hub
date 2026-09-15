import torch
from nnsight import LanguageModel

from jlens.lens import JacobianLens
from jlens.vis import _meaningful_token_mask, _ranks_of


def jlens_head_by_mul(
    nn_model: LanguageModel, 
    lens: JacobianLens,
    layer: int, 
    head: int, 
):
    """
    Applies W_U @ J @ W_x formulation to characterize an attention head.
    """
    d_model = nn_model.config.hidden_size
    n_heads = nn_model.config.num_attention_heads
    n_kv_heads = getattr(nn_model.config, "num_key_value_heads", n_heads)
    print(f"d_model={d_model} n_heads={n_heads} n_kv_heads={n_kv_heads}")

    head_dim = d_model // n_heads
    kv_groups = n_heads // n_kv_heads
    kv_h = head // kv_groups    # k/v head index in GQA models
    print(f"head_dim={head_dim} kv_groups={kv_groups} kv_h={kv_h}")

    token_scores = {}
    for name in ["W_Q", "W_K", "W_V", "W_O^T"]:
        J_matrix = lens.jacobians[layer] if name == "W_O^T" else lens.jacobians[layer-1]
        with nn_model.session(remote=True):
            attn = nn_model.model.layers[layer].self_attn
            if name == "W_Q":
                # dim: [d_model, head_dim]
                w = attn.q_proj.weight[head * head_dim : (head + 1) * head_dim, :].T.save()
            elif name == "W_K":
                w = attn.k_proj.weight[kv_h * head_dim : (kv_h + 1) * head_dim, :].T.save()
            elif name == "W_V":
                w = attn.v_proj.weight[kv_h * head_dim : (kv_h + 1) * head_dim, :].T.save()
            else:    # name == "W_O^T"
                w = attn.o_proj.weight[:, head * head_dim : (head + 1) * head_dim].save()

        residual = torch.matmul(J_matrix, w.float().cpu())
        with nn_model.session(remote=True):
            scores = torch.matmul(nn_model.lm_head.weight, residual.to("cuda:0")).norm(dim=1).save()
        token_scores[name] = scores
        print(f"{name}_scores.shape={scores.shape}")
    return token_scores


def jlens_head_by_fw(
    nn_model: LanguageModel, 
    lens: JacobianLens,
    layer: int, 
    head: int, 
):
    """
    Applies normal forward pass to characterize an attention head.
    """
    d_model = nn_model.config.hidden_size
    n_heads = nn_model.config.num_attention_heads
    n_kv_heads = getattr(nn_model.config, "num_key_value_heads", n_heads)
    print(f"d_model={d_model} n_heads={n_heads} n_kv_heads={n_kv_heads}")

    head_dim = d_model // n_heads
    kv_groups = n_heads // n_kv_heads
    kv_h = head // kv_groups    # k/v head index in GQA models
    print(f"head_dim={head_dim} kv_groups={kv_groups} kv_h={kv_h}")

    token_scores = {}
    for name in ["W_Q", "W_K", "W_V", "W_O^T"]:
        J_matrix = lens.jacobians[layer] if name == "W_O^T" else lens.jacobians[layer-1]
        with nn_model.session(remote=True):
            attn = nn_model.model.layers[layer].self_attn
            if name == "W_Q":
                # dim: [d_model, head_dim]
                w = attn.q_proj.weight[head * head_dim : (head + 1) * head_dim, :].T.save()
            elif name == "W_K":
                w = attn.k_proj.weight[kv_h * head_dim : (kv_h + 1) * head_dim, :].T.save()
            elif name == "W_V":
                w = attn.v_proj.weight[kv_h * head_dim : (kv_h + 1) * head_dim, :].T.save()
            else:    # name == "W_O^T"
                w = attn.o_proj.weight[:, head * head_dim : (head + 1) * head_dim].save()

        residual = torch.matmul(J_matrix, w.float().cpu())
        with nn_model.session(remote=True):
            normed = nn_model.model.norm(residual.T.to("cuda:0").to(nn_model.dtype))
            logits = nn_model.lm_head(normed)
            logit_softcap = getattr(nn_model.config, "final_logit_softcapping", None)
            if logit_softcap is not None:
                logits = logit_softcap * torch.tanh(logits / logit_softcap)
            scores = logits.norm(dim=0).save()
        token_scores[name] = scores
        print(f"{name}_scores.shape={scores.shape}")
    return token_scores


def rank_tokens(tokenizer, token_scores, layer, head, top_k):
    results = {}
    for name, scores in token_scores.items():     
        vocab_size = scores.shape[0]           
        display_mask = _meaningful_token_mask(tokenizer, vocab_size, scores.device)
        top_idx = scores.masked_fill(~display_mask, float("-inf")).topk(top_k).indices
        top_ranks = _ranks_of(scores.unsqueeze(0), top_idx)[0]
        results[name] = [f"{tokenizer.decode(idx.item()).strip()} ({top_ranks[i]})" for i, idx in enumerate(top_idx)]

    print(f"\nJ-Lens Readout | Layer {layer} | Head {head}")
    print(f"{'W_Q (rank)':<25} {'W_K (rank)':<25} {'W_V (rank)':<25} {'W_O^T (rank)':<25}")    
    for i in range(top_k):
        print(f"{results['W_Q'][i]:<25} {results['W_K'][i]:<25} {results['W_V'][i]:<25} {results['W_O^T'][i]:<25}")
