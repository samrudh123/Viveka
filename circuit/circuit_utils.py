import os
import warnings

from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
import ipywidgets as widgets

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CAUTION: for efficiency, automatic differentiation is being disabled globally in PyTorch
# Shouldn't matter unless you want to finetune
torch.set_grad_enabled(False)

def extract_activations(model: HookedTransformer, prompt: str, filename: Optional[str] = None,
                        max_new_tokens: int = 20, temperature: float = 0) -> Dict[str, Any]:
    """
    Runs the model on the input text, caches activations, and returns results.

    Parameters
    ----------
    model : HookedTransformer
        The transformer model to run.
    prompt : str
        Input text prompt.
    filename : str or None, optional
        File path to save the cache. If None, no file is saved.
    max_new_tokens : int, optional
        Maximum number of tokens to generate.
    temperature : float, optional
        Temperature of the model. Set to 0 (greedy decoding) by default.

    Returns
    -------
    dict
        Dictionary containing:
        - 'activations': cached intermediate values.
        - 'final_logits': output logits.
        - 'str_tokens': list of token strings.
    """

    # Generate model response tokens
    tokens = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prepend_bos=False,
        return_type='tokens'
    )

    # Run model and record activations cache
    final_logits, cache = model.run_with_cache(tokens, prepend_bos=False)
    cache = cache.remove_batch_dim()
    cache_data = {
        'activations': cache.cpu() if filename is not None else cache,
        'final_logits': final_logits.cpu() if filename is not None else final_logits,
        'str_tokens': model.to_str_tokens(tokens[0])
    }

    # Save cache data to disk if filename is provided
    if filename is not None:
        save_path = f'{filename}.pt'
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(cache_data, save_path)

    return cache_data

# =====================================================================================
# LOGIT LENS
# =====================================================================================

def logit_lens(model: HookedTransformer, activations: Float[torch.Tensor, '... d_model'],
               norm: bool = False, p: float = 2)-> Float[torch.Tensor, '... vocab_size']:
    """
    Projects hidden activations to vocabulary logits using final LayerNorm and unembedding.

    Parameters
    ----------
    model : HookedTransformer
        The model providing projection layers.
    activations : torch.Tensor
        Tensor of shape [..., d_model] representing intermediate activations.
    norm : bool
        Whether to normalize the unembedding matrix or not (for NormedLens)
    p : float
        The parameter for normalizing the unembedding matrix using p-norm.
        Considered only if `norm=True`

    Returns
    -------
    torch.Tensor
        Logits over vocabulary, shape [..., vocab_size].
    """
    # Move activations to the model's device if needed
    if activations.device != model.cfg.device:
        activations = activations.to(model.cfg.device)

    # Apply layer normalization and unembedding
    normalized = model.ln_final(activations)
    if norm:
        logits = normalized @ F.normalize(model.W_U, p=p, dim=0)
    else:
        logits = model.unembed(normalized)

    return logits

def plot_top_k(model: HookedTransformer, cache_data: Dict, hook_name: str,
               token_pos: int, k: int = 10, norm: bool = False, p: float = 2):
    """
    Visualizes top-k predicted tokens and their probabilities for a given layer/hook and token position.

    Parameters
    ----------
    model : HookedTransformer
        The transformer model.
    cache_data : dict
        Cached activations and string tokens.
    hook_name : str
        Activation hook name to use.
    token_pos : int
        Token position in the sequence.
    k : int, optional
        Number of top predictions to show.
    norm : bool
        Whether to normalize the unembedding matrix or not (for NormedLens)
    p : float
        The parameter for normalizing the unembedding matrix using p-norm.
        Considered only if `norm=True`
    """
    # Retrieve activations for the specified hook and token position
    activations = cache_data['activations'][hook_name][token_pos]
    logits = logit_lens(model, activations, norm, p)

    # Get top-k logits and their indices
    top_logits, top_indices = torch.topk(logits, k=k)
    probs = F.softmax(logits, dim=-1)[top_indices].tolist()
    tokens = [model.to_string(idx.item()) for idx in top_indices]

    # Create Plotly bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(tokens))),
        y=probs,
        text=[f"'{token}'" for token in tokens],
        textposition='outside',
        hovertemplate='<b>Token:</b> %{text}<br><b>Probability:</b> %{y:.4f}<extra></extra>',
        marker=dict(
            color=probs,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Probability")
        )
    ))
    fig.update_layout(
        title=f'Top-{k} Predictions at Position {token_pos}<br>'
              f'Layer {hook_name.split(".")[1]}, Hook: {hook_name}<br>'
              f'Current Token: \"{cache_data["str_tokens"][token_pos]}\"',
        xaxis_title='Rank',
        yaxis_title='Probability',
        xaxis=dict(tickmode='array', tickvals=list(range(len(tokens))),
                  ticktext=[f'{i+1}' for i in range(len(tokens))]),
        height=500,
        showlegend=False
    )
    fig.show()

    # Print summary
    print(f"\nHook: {hook_name}")
    print(f"Token Position: {token_pos}")
    print(f"Current Token: '{cache_data['str_tokens'][token_pos]}'")
    print(f"Top prediction: '{tokens[0]}' (prob: {probs[0]:.4f})")

def create_interactive_widget(model: HookedTransformer, cache_data: Dict,
                              norm: bool = False, p: float = 2):
    """
    Creates an interactive widget for exploring logit lens predictions across layers and tokens.

    Parameters
    ----------
    model : HookedTransformer
        The transformer model.
    cache_data : dict
        Cached activations and tokens.
    norm : bool
        Whether to normalize the unembedding matrix or not (for NormedLens)
    p : float
        The parameter for normalizing the unembedding matrix using p-norm.
        Considered only if `norm=True`
    """
    activations = cache_data['activations']

    # Get available layers and hooks on which logit lens can be done
    layer_hooks = {}
    for key in activations.keys():
        if key.startswith('blocks.') and ('resid' in key or 'normalized' in key or 'out' in key):
            layer = key.split('.')
            layer_num = int(layer[1])
            if layer_num not in layer_hooks:
                layer_hooks[layer_num] = []
            layer_hooks[layer_num].append('.'.join(layer[2:]))

    # Get tokens and its length
    str_tokens = cache_data['str_tokens']
    seq_len = len(str_tokens)

    # Create widgets
    layer_slider = widgets.IntSlider(value=0, min=0, max=len(layer_hooks)-1, step=1,
                                     description='Layer:', style={'description_width': 'initial'})

    hook_dropdown = widgets.Dropdown(
        options=layer_hooks[sorted(layer_hooks.keys())[0]],
        value=layer_hooks[sorted(layer_hooks.keys())[0]][0],
        description='Hook:',
        style={'description_width': 'initial'}
    )

    position_slider = widgets.IntSlider(value=min(16, seq_len-1), min=0, max=seq_len-1,
                                        step=1, description='Token Position:',
                                        style={'description_width': 'initial'})

    k_slider = widgets.IntSlider(value=10, min=1, max=20, step=1, description='Top-K:',
                                 style={'description_width': 'initial'})

    # Update hook dropdown when layer changes
    def update_hook_options(change):
        layer = change['new']
        if layer in layer_hooks:
            value = hook_dropdown.value
            hook_dropdown.options = layer_hooks[layer]
            hook_dropdown.value = value if value in layer_hooks[layer] else layer_hooks[layer][0]

    layer_slider.observe(update_hook_options, names='value')

    # Create interactive plot
    def update_plot(layer, hook_name, token_position, k):
        plot_top_k(model, cache_data, f'blocks.{layer}.{hook_name}', token_position, k, norm, p)

    # Display widgets
    widgets.interact(update_plot, layer=layer_slider, hook_name=hook_dropdown,
                     token_position=position_slider, k=k_slider)

    # Show sequence for reference
    print(f"\nSequence ({seq_len} tokens):")
    for i, token in enumerate(str_tokens):
        print(f"{i:2d}: '{token}'")

def display_attention_patterns(
    model: HookedTransformer,
    cache_data: Dict,
    layer_num: int,
    hook_name: str,
    token_range: Optional[Tuple[int, int]] = None
):
    """
    Displays attention head patterns for a given layer using different weighting schemes.

    Parameters
    ----------
    model : HookedTransformer
        The model to extract patterns from.
    cache_data : dict
        Dictionary containing cached tensors and tokens.
    layer_num : int
        Layer index to visualize.
    hook_name : {'hook_pattern', 'hook_z', 'hook_attn_out'}
        Type of attention visualization.
    token_range : tuple of int, optional
        Start and end indices of the token window to visualize.
    """
    str_tokens = cache_data['str_tokens']
    seq_len = len(str_tokens)

    # Determine token range to visualize
    start_pos, end_pos = (0, seq_len) if token_range is None else (
        max(0, token_range[0]), min(seq_len, token_range[1])
    )

    # Model and layer parameters
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    device = model.cfg.device

    # Get hook names for attention pattern and value vectors
    pattern_hook = f'blocks.{layer_num}.attn.hook_pattern'
    z_hook = f'blocks.{layer_num}.attn.hook_z'

    # Retrieve and move tensors to device
    attn_pattern = cache_data['activations'][pattern_hook].to(device)  # [n_heads, seq, seq]
    z_values = cache_data['activations'][z_hook].to(device)            # [seq, n_heads * d_head]

    # Choose visualization type
    if hook_name == 'hook_pattern':
        # Use raw attention pattern
        pattern_to_plot = attn_pattern[:, start_pos:end_pos, start_pos:end_pos]
        title_suffix = "Attention Pattern"

    elif hook_name == 'hook_z':
        # Weight attention pattern by norm of value vectors (z)
        z_reshaped = z_values.view(-1, n_heads, d_head)
        pattern_to_plot = torch.zeros(n_heads, end_pos - start_pos, end_pos - start_pos, device=device)

        for head in range(n_heads):
            head_pattern = attn_pattern[head, start_pos:end_pos, start_pos:end_pos]
            head_z = z_reshaped[start_pos:end_pos, head, :]                    # [range_len, d_head]
            z_magnitudes = torch.norm(head_z, dim=-1)                          # [range_len]

            weighted_pattern = head_pattern * z_magnitudes[None, :]           # [range_len, range_len]
            pattern_to_plot[head] = weighted_pattern

        title_suffix = "Value-Weighted Attention Pattern"

    elif hook_name == 'hook_attn_out':
        # Weight attention pattern by the magnitude of head output vectors (z @ W_O)
        W_O = model.blocks[layer_num].attn.W_O                                # [n_heads, d_head, d_model]
        z_reshaped = z_values.view(-1, n_heads, d_head)
        pattern_to_plot = torch.zeros(n_heads, end_pos - start_pos, end_pos - start_pos, device=device)

        for head in range(n_heads):
            head_pattern = attn_pattern[head, start_pos:end_pos, start_pos:end_pos]
            head_z = z_reshaped[start_pos:end_pos, head, :]                   # [range_len, d_head]
            head_output = torch.matmul(head_z, W_O[head])                     # [range_len, d_model]
            output_magnitudes = torch.norm(head_output, dim=-1)              # [range_len]

            weighted_pattern = head_pattern * output_magnitudes[:, None]     # [range_len, range_len]
            pattern_to_plot[head] = weighted_pattern

        title_suffix = "Individual Head Effects (z @ W_O weighted)"

    else:
        raise ValueError(f"Unsupported hook name: '{hook_name}'. Choose from 'hook_pattern', 'hook_z', or 'hook_attn_out'.")

    # Setup for plotting
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Head {i}' for i in range(n_heads)],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )

    token_labels = [f"{i}: '{str_tokens[i]}'" for i in range(start_pos, end_pos)]

    # Plot each head's pattern
    for head in range(n_heads):
        row, col = divmod(head, cols)
        pattern = pattern_to_plot[head].detach().cpu().numpy()

        fig.add_trace(
            go.Heatmap(
                z=pattern,
                x=token_labels,
                y=token_labels,
                colorscale='Viridis',
                showscale=(head == 0),  # Only show color scale on first subplot
                hovertemplate='From: %{y}<br>To: %{x}<br>Value: %{z:.4f}<extra></extra>'
            ),
            row=row + 1,
            col=col + 1
        )

    # Final layout settings
    fig.update_layout(
        title=f'Layer {layer_num} - {title_suffix}<br>Token Range: {start_pos} to {end_pos - 1}',
        height=300 * rows,
        showlegend=False
    )

    # Label axes
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(title_text="To Token", row=i, col=j)
            fig.update_yaxes(title_text="From Token", row=i, col=j)

    fig.show()

    # Console summary
    print(f"\nDisplaying {title_suffix}")
    print(f"Layer: {layer_num}")
    print(f"Token range: {start_pos} to {end_pos - 1}")
    print(f"Number of heads: {n_heads}")

def create_attention_widget(model: HookedTransformer, cache_data: Dict):
    """
    Launches an interactive widget to visualize attention patterns by layer, token range, and hook.

    Parameters
    ----------
    model : HookedTransformer
        The transformer model.
    cache_data : dict
        Dictionary with cached activations and tokens.
    """
    # Get available layers
    available_layers = []
    for key in cache_data['activations'].keys():
        if 'blocks.' in key and 'attn' in key:
            try:
                layer_num = int(key.split('.')[1])
                if layer_num not in available_layers:
                    available_layers.append(layer_num)
            except:
                continue

    available_layers.sort()

    if not available_layers:
        raise ValueError('No attention layers available!')

    # Get tokens and its length
    str_tokens = cache_data['str_tokens']
    seq_len = len(str_tokens)

    # Create widgets
    layer_slider = widgets.IntSlider(value=0, min=0, max=len(available_layers)-1, step=1,
                                     description='Layer:', style={'description_width': 'initial'})

    hook_dropdown = widgets.Dropdown(
        options=['hook_pattern', 'hook_z', 'hook_attn_out'],
        value='hook_pattern',
        description='Hook Type:',
        style={'description_width': 'initial'}
    )

    start_slider = widgets.IntSlider(value=0, min=0, max=seq_len-1, step=1,
                                     description='Start Position:',
                                     style={'description_width': 'initial'})

    end_slider = widgets.IntSlider(value=min(10, seq_len), min=1, max=seq_len,
                                   step=1, description='End Position:',
                                   style={'description_width': 'initial'})

    # Ensure end > start
    def update_end_min(change):
        end_slider.min = start_slider.value + 1
        if end_slider.value <= start_slider.value:
            end_slider.value = start_slider.value + 1

    def update_start_max(change):
        start_slider.max = end_slider.value - 1
        if start_slider.value >= end_slider.value:
            start_slider.value = end_slider.value - 1

    start_slider.observe(update_end_min, names='value')
    end_slider.observe(update_start_max, names='value')

    # Create interactive plot
    def update_plot(layer, hook_name, start_pos, end_pos):
        token_range = (start_pos, end_pos)
        display_attention_patterns(model, cache_data, layer, hook_name, token_range)

    # Display widgets
    widgets.interact(
        update_plot,
        layer=layer_slider,
        hook_name=hook_dropdown,
        start_pos=start_slider,
        end_pos=end_slider
    )

    # Show sequence for reference
    print(f"\nFull Sequence ({seq_len} tokens):")
    for i, token in enumerate(str_tokens):
        print(f"{i:2d}: '{token}'")

def plot_logit_lens_heatmap(
    model: HookedTransformer,
    cache_data: Dict,
    start: int,
    hook_filter: Union[str, List[str]],
    norm: bool = False,
    p: float = 2,
    figsize: Tuple[int, int] = (35, 12)
):
    """
    Plots a heatmap of top logit lens predictions across layers and token positions.

    Parameters
    ----------
    model : HookedTransformer
        The model used for projecting logits.
    cache_data : dict
        Cached activations and tokens.
    start : int
        Starting token index.
    hook_filter : str or list of str
        Filters for selecting relevant layer hooks. Should be a substring or a list of
        substrings that will be matched against activation hook names on which logit
        lens can be done. For example, 'attn' will pick the layers of the form
        `blocks.{layer}.hook_attn_out`, 'resid_post' will pick the layers of the form
        `blocks.{layer}.hook_resid_post`, ['attn', 'mlp'] will pick the layers of both
        the forms `blocks.{layer}.hook_attn_out` and `blocks.{layer}.hook_mlp_out`
    norm : bool
        Whether to normalize the unembedding matrix or not (for NormedLens)
    p : float
        The parameter for normalizing the unembedding matrix using p-norm.
        Considered only if `norm=True`
    figsize : tuple of int, optional
        Size of the plot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure showing top token predictions and their confidences.
    """

    # Normalize hook_filter to a list
    if isinstance(hook_filter, str):
        hook_filter = [hook_filter]

    # Retrieve token data and activations
    str_tokens = cache_data['str_tokens']
    seq_len = len(str_tokens)
    activations_data = cache_data['activations']

    # Identify relevant hooks
    hooks = []
    for key in sorted(
        activations_data.keys(),
        key=lambda x: int(x.split('.')[1]) if 'blocks.' in x else float('inf')
    ):
        if (
            key.startswith('blocks.')
            and ('resid' in key or 'normalized' in key or 'out' in key)
            and any(f in key for f in hook_filter)
        ):
            hooks.append(key)

    if not hooks:
        raise ValueError('No hooks under the given filter!')

    # Collect top predictions and probabilities
    all_preds = []
    all_probs = []

    for hook in hooks:
        activations = activations_data[hook]  # shape: [seq_len, d_model]
        logits = logit_lens(model, activations, norm, p)  # shape: [seq_len, vocab_size]

        probs = F.softmax(logits, dim=-1)  # shape: [seq_len, vocab_size]
        if norm:
            probs = torch.log(probs)
        top_probs, top_indices = torch.max(probs, dim=-1)  # shape: [seq_len]

        top_tokens = model.to_str_tokens(top_indices, prepend_bos=False)

        all_preds.append(top_tokens)
        all_probs.append(top_probs.detach().cpu().numpy())

    # Slice to target token range
    pred_matrix = np.array(all_preds, dtype=object)[:, start:]
    prob_matrix = np.array(all_probs)[:, start:]

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        prob_matrix,
        annot=pred_matrix,
        fmt='',
        cmap='viridis',
        cbar_kws={'label': f'Prediction {"Log-p" if norm else "P"}robability'},
        xticklabels=[f'Pos {i}' for i in range(start, seq_len)],
        yticklabels=[f"Layer {' '.join(hook.split('.')[1:])}" for hook in hooks],
        ax=ax
    )

    # Add actual tokens above the plot
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(len(str_tokens[start:])) + 0.5)
    ax2.set_xticklabels(str_tokens[start:], rotation=45, ha='left')
    ax2.set_xlabel('Actual Tokens')

    # Set labels and title
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    ax.set_title(f'Logit Lens{f" ({p}-Normed)" if norm else ""}: Top Predictions Across Layers')

    # Avoid font warnings
    warnings.filterwarnings("ignore", message="Glyph.*missing from current font")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

    return fig

def plot_token_rank_heatmap(
    model: HookedTransformer,
    cache_data: Dict,
    tokens: Union[str, List[str]],
    start: int,
    hook_filter: Union[str, List[str]],
    norm: bool = False,
    p: float = 2,
    figsize: Tuple[int, int] = (35, 12)
):
    """
    Visualizes the rank of specific tokens at each layer/token position using the logit lens.

    Parameters
    ----------
    model : HookedTransformer
        The transformer model.
    cache_data : dict
        Dictionary with activations and tokens.
    tokens : str or list of str
        Target tokens to track.
    start : int
        Start index of the token sequence.
    hook_filter : str or list of str
        Filters for selecting relevant layer hooks. Should be a substring or a list of
        substrings that will be matched against activation hook names on which logit
        lens can be done. For example, 'attn' will pick the layers of the form
        `blocks.{layer}.hook_attn_out`, 'resid_post' will pick the layers of the form
        `blocks.{layer}.hook_resid_post`, ['attn', 'mlp'] will pick the layers of both
        the forms `blocks.{layer}.hook_attn_out` and `blocks.{layer}.hook_mlp_out`
    norm : bool
        Whether to normalize the unembedding matrix or not (for NormedLens)
    p : float
        The parameter for normalizing the unembedding matrix using p-norm.
        Considered only if `norm=True`
    figsize : tuple of int, optional
        Size of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        Heatmap showing token ranks across layers.
    """

    # Normalize hook_filter and tokens to a list
    if isinstance(hook_filter, str):
        hook_filter = [hook_filter]
    if isinstance(tokens, str):
        tokens = [tokens]

    # Retrieve token data and activations
    str_tokens = cache_data['str_tokens']
    seq_len = len(str_tokens)
    activations_data = cache_data['activations']

    # Identify relevant hooks
    hooks = []
    for key in sorted(
        activations_data.keys(),
        key=lambda x: int(x.split('.')[1]) if 'blocks.' in x else float('inf')
    ):
        if (
            key.startswith('blocks.')
            and ('resid' in key or 'normalized' in key or 'out' in key)
            and any(f in key for f in hook_filter)
        ):
            hooks.append(key)

    # Error handling
    if not hooks:
        raise ValueError('No hooks under the given filter!')
    
    try:
        token_ids = [model.to_single_token(tok) for tok in tokens]
        token_tensor = torch.tensor(token_ids, device=model.cfg.device)
    except AssertionError:
        raise ValueError("One or more tokens could not be converted to a single token.")

    # Collect top-1 predictions and probabilities
    annotations = []
    rank_values = []

    for hook in hooks:
        activations = activations_data[hook]  # shape: [seq_len, d_model]
        logits = logit_lens(model, activations, norm, p)  # shape: [seq_len, vocab_size]

        # Computing the rank and annotations
        sorted_indices = logits.argsort(dim=-1, descending=True)
        rank_map = sorted_indices.argsort(dim=-1)
        selected_ranks = rank_map[:, token_tensor]
        ranks, min_pos = selected_ranks.min(dim=1)
        top_tokens = model.to_str_tokens(token_tensor[min_pos], prepend_bos=False)

        annotations.append([f'{token}: {rank.item()}' for token, rank in zip(top_tokens, ranks)])
        rank_values.append((ranks).detach().cpu().numpy())

    # Slice to target token range
    annot_matrix = np.array(annotations, dtype=object)[:, start:]
    rank_matrix = np.array(rank_values)[:, start:]

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        rank_matrix,
        annot=annot_matrix,
        fmt='',
        cmap='viridis',
        cbar_kws={'label': 'Rank'},
        xticklabels=[f'Pos {i}' for i in range(start, seq_len)],
        yticklabels=[f"Layer {' '.join(hook.split('.')[1:])}" for hook in hooks],
        ax=ax
    )

    # Add actual tokens above the plot
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(len(str_tokens[start:])) + 0.5)
    ax2.set_xticklabels(str_tokens[start:], rotation=45, ha='left')
    ax2.set_xlabel('Actual Tokens')

    # Set labels and title
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    ax.set_title('Best rank per position among tokens')

    # Avoid font warnings
    warnings.filterwarnings("ignore", message="Glyph.*missing from current font")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

    return fig

def create_head_contribution_widget(model: HookedTransformer, cache_data: Dict,token: str):
    '''
    Plots contribution of each attention head calculated by the dot product of head's
    output (after applying final layer norm) with the token's residual stream direction.

    Parameters
    ----------
    model: HookedTransformer
        The transformer model.
    cache_data : dict
        Dictionary with activations and tokens.
    token : str
        Target token.
    '''
    # Token processing
    str_tokens = cache_data['str_tokens']
    seq_len = len(str_tokens)
    resid_dir = model.tokens_to_residual_directions(token)

    # Attention head outputs
    head_out = cache_data['activations'].stack_head_results()
    head_out = model.ln_final(head_out)
    dot = torch.einsum('d,ctd->tc', resid_dir, head_out)
    matrix = dot.reshape(seq_len, model.cfg.n_layers, model.cfg.n_heads)
    
    token_slider = widgets.IntSlider(value=0, min=0, max=seq_len-1, step=1,
                                     description='Start Position:',
                                     style={'description_width': 'initial'})
    
    # Create interactive plot
    def update_plot(pos):
        px.imshow(
            matrix[pos].detach().cpu().numpy(),
            color_continuous_midpoint=0.0,
            color_continuous_scale="RdBu",
            labels={"x": "Head", "y": "Layer"},
            title=f'Contribution to token "{token}" above token "{str_tokens[pos]}"',
        ).show()

    # Display widgets
    widgets.interact(update_plot, pos=token_slider)

# =====================================================================================
# CAUSAL TRACING TOOLS
# =====================================================================================

Hook = Tuple[int, Callable]

def run_with_hooks(
    model: HookedTransformer,
    clean_cache: Dict,
    hooks: List[Hook],
    filename: Optional[str] = None
) -> Dict[str, Any]:
    '''
    Runs a forward pass through a HookedTransformer model starting from a specified
    middle layer, using cached activations to avoid redundant lower-layer computations,
    and applies user-defined hooks.

    Parameters
    ----------
    model: HookedTransformer
        The transformer model
    clean_cache: Dict
        The dictionary containing activation cache providing precomputed activations of the
        prompt without any hooks, the final logits and str_tokens of the prompt, typically
        generated using `extract_activations()` function
    hooks: list[Hook]
        The hooks to apply to the model. Typically generated using the following functions
        - ablate(token, layer, kind)
        - patch(token, layer, kind, patch_cache)
        - scale_attn_scores(layer, scaling_factor, head)
        - remove_direction(act_name, direction)
    filename : str or None, optional
        File path to save the cache. If None, no file is saved.

    Returns
    -------
    dict
        Dictionary containing:
        - 'activations': cached intermediate values.
        - 'final_logits': output logits.
        - 'str_tokens': list of token strings.
    '''
    # Getting the cache-storing hooks
    cache_dict, fwd_hooks, _ = model.get_caching_hooks()

    # Adding additional hooks
    resume_layer = model.cfg.n_layers - 1
    hook_list = []
    for hook_name, hook_fn in hooks:
        layer = int(hook_name.split('.')[1])
        assert layer < model.cfg.n_layers
        if layer < resume_layer:
            resume_layer = layer
        hook_list.append((hook_name, hook_fn))

    # Running the model in an optimized way
    with model.hooks(hook_list + fwd_hooks) as hooked_model:
        logits = hooked_model.forward(
            clean_cache['activations'][f'blocks.{resume_layer}.hook_resid_pre'].unsqueeze(0),
            start_at_layer=resume_layer
        )

    # Updating cache activations
    cache = dict(clean_cache['activations'])
    for key in cache_dict:
        cache_dict[key] = cache_dict[key].squeeze(0)
    cache.update(cache_dict)
    cache = ActivationCache(cache, model, has_batch_dim=False)

    # Preparing cache_data
    cache_data = {
        'activations': cache.cpu() if filename is not None else cache,
        'final_logits': logits.cpu() if filename is not None else logits,
        'str_tokens': clean_cache['str_tokens']
    }

    # Save cache data to disk if filename is provided
    if filename is not None:
        save_path = f'{filename}.pt'
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(cache_data, save_path)

    return cache_data

def ablate(token: int, layer: int, kind: Literal['attn', 'mlp']) -> Hook:
    """
    Ablates attention/MLP block output into the residual stream at a specified token position and layer.

    Parameters
    ----------
    token : int
        Token position to apply the ablation
    layer : int
        Layer number to apply the ablation (indexed from 0)
    kind : {'attn', 'mlp'}
        Kind of layer to apply ablation on
        - 'attn': to apply on hook_attn_out
        - 'mlp': to apply on hook_mlp_out
    """
    def hook_fn(activations: Float[torch.Tensor, 'batch pos d_model'], hook: HookPoint):
        activations[:, token, :] = 0
        return activations
    return f'blocks.{layer}.hook_{kind}_out', hook_fn

def patch(
    token: int,
    layer: int,
    kind: Literal['attn', 'mlp', 'pre', 'mid', 'post'],
    patch_cache: ActivationCache,
    patch_token: Optional[int] = None
) -> Hook:
    """
    Patches specified activation with that from of patch_cache at a specified token position and layer

    Parameters
    ----------
    token : int
        Token position to apply the ablation
    layer : int
        Layer number to apply the ablation (indexed from 0)
    kind : {'attn', 'mlp', 'pre', 'mid', 'post'}
        Kind of layer to apply ablation on
        - 'attn': to apply on hook_attn_out
        - 'mlp': to apply on hook_mlp_out
        - 'pre': to apply on hook_resid_pre
        - 'mid': to apply on hook_resid_mid
        - 'post': to apply on hook_resid_post
    patch_cache : ActivationCache
        The activation cache whose activation will be used for patching
    patch_token : int or None, optional
        The token position to patch from patch_cache. Defaults to the same value as `token`
    """
    if kind in ['attn', 'mlp']:
        kind = 'hook_' + kind + '_out'
    elif kind in ['pre', 'mid', 'post']:
        kind = 'hook_resid_' + kind
    else:
        raise ValueError("kind should be one of ['attn', 'mlp', 'pre', 'mid', 'post']")
    
    if patch_token is None:
        patch_token = token

    def hook_fn(activations: Float[torch.Tensor, 'batch pos d_model'], hook: HookPoint):
        activations[:, token, :] = patch_cache[hook.name][patch_token, :]
        return activations
    return f'blocks.{layer}.{kind}', hook_fn

def scale_attn_scores(layer: int, scaling_factor: float, head: Optional[int] = None) -> Hook:
    '''
    Scales attention scores before softmax to amplify or flatten the scores

    Parameters
    ----------
    layer : int
        Layer number to scale the attention scores (indexed from 0)
    scaling_factor : float
        The factor by which to multiply the attention score matrix
        - \< 1: for flattening ie. making attention pattern more uniform
        - \> 1: for amplifying ie. making attention pattern more peaked
    head : int or None, optional
        The head index to apply the scaling. If None, applies to all heads. Defaults to None.
    '''
    def hook_fn(attn_scores: Float[torch.Tensor, 'batch head from_pos to_pos'], hook: HookPoint):
        if head is None:
            attn_scores *= scaling_factor
        else:
            assert head < attn_scores.shape[1]
            attn_scores[:, head, :, :] *= scaling_factor
        return attn_scores
    return f'blocks.{layer}.attn.hook_attn_scores', hook_fn

def attn_knockout(layer: int, from_tok: int, to_tok: int, head: Optional[int] = None) -> Hook:
    '''
    Disables `from_tok` to `to_tok` attention interaction in a particular layer

    Parameters
    ----------
    layer : int
        Layer number to apply the knockout (indexed from 0)
    from_tok : int
        Source token position
    to_tok : int
        Destination token position
    head : int or None, optional
        The head index to apply the scaling. If None, applies to all heads. Defaults to None.
    '''
    def hook_fn(attn_scores: Float[torch.Tensor, 'batch head from_pos to_pos'], hook: HookPoint):
        assert from_tok < attn_scores.shape[2]
        assert to_tok < attn_scores.shape[3]
        if head is None:
            attn_scores[:, :, from_tok, to_tok] = 0
        else:
            assert head < attn_scores.shape[1]
            attn_scores[:, head, from_tok, to_tok] = 0
        return attn_scores
    return f'blocks.{layer}.attn.hook_attn_scores', hook_fn

def remove_direction(act_name: str, direction: Float[torch.Tensor, 'd_model']) -> Hook:
    '''
    Removes component of residual stream along direction in the layer specified by activation_name for the last token
    
    Parameters
    ----------
    act_name : str
        Layer name to apply direction removal.
        Use transformer_lens.utils.get_act_name() for more help.
    direction : torch.Tensor
        Tensor of shape [d_model] representing the direction to remove (should be on same device).
    '''
    direction /= direction.norm()
    def hook_fn(activations: Float[torch.Tensor, 'batch pos d_model'], hook: HookPoint):
        activations[:, -1, :] -= activations[0, -1, :].dot(direction) * direction
        return activations
    return act_name, hook_fn

def plot_patching_experiment(
    model: HookedTransformer,
    cache1: Dict[str, Any],
    cache2: Dict[str, Any],
    word_token1: Optional[str] = None,
    word_token2: Optional[str] = None,
    start: int = 0,
    end: int = None,
    target_pos: Optional[int] = None,
    kind: Literal['attn', 'mlp', 'pre', 'mid', 'post'] = 'post'
):
    """
    Conducts a patching experiment, replacing activations from cache1 with those from cache2
    at each layer and token position, and visualizes the logit difference between two target
    tokens as an interactive Plotly heatmap.

    Parameters
    ----------
    model : HookedTransformer
        The transformer model.
    cache1 : dict
        Baseline activation cache (from extract_activations).
    cache2 : dict
        Patch activation cache (from extract_activations).
    word_token1 : str, optional
        First target token for logit difference.
        If None, defaults to the first token predicted by the model in cache1
    word_token2 : str, optional
        Second target token for logit difference.
        If None, defaults to the first token predicted by the model in cache2
    start : int, optional
        Token position to start the patching experiment, by default 0.
    end : int, optional
        Token position to end the patching experiment. (This token won't be patched.)
        If None, defaults to the last token position.
    target_pos : int, optional
        Token position where to check for logit difference.
        By default, it is the position just after the token 'model'.
    kind : {'attn', 'mlp', 'pre', 'mid', 'post'}
        Kind of layer to patch, by default 'post'.
    """
    n_layers = model.cfg.n_layers
    seq_len = len(cache1['str_tokens'])

    target_pos = cache1['str_tokens'].index('model') + 1 if target_pos is None else target_pos
    word_token1 = word_token1 or cache1['str_tokens'][target_pos + 1]
    word_token2 = word_token2 or cache2['str_tokens'][target_pos + 1]
    token_id1 = model.to_single_token(word_token1)
    token_id2 = model.to_single_token(word_token2)

    if end is None:
        end = target_pos + 1
    assert end <= seq_len
    assert start < end

    logit_diffs = np.zeros((n_layers, end - start))
    for layer in range(n_layers):
        for pos in range(start, end):
            hook = patch(token=pos, layer=layer, kind=kind, patch_cache=cache2['activations'])
            patched_cache = run_with_hooks(model, cache1, hooks=[hook])
            logits = patched_cache['final_logits'][0][target_pos]
            logit_diffs[layer, pos - start] = (logits[token_id1] - logits[token_id2]).item()

    # Prepare axis and hover labels
    layer_labels = [f'Layer {i}' for i in range(n_layers)]
    pos_labels = [f'{i}: {tok1} --> {tok2}'
                  for i, (tok1, tok2) in enumerate(
                      zip(
                          cache1['str_tokens'][start:end],
                          cache2['str_tokens'][start:end]
                      ), start=start
                  )]

    fig = px.imshow(
        logit_diffs,
        labels={
            'x': 'Token Position',
            'y': 'Layer',
            'color': f'Logit difference: "{word_token1}" - "{word_token2}"'
        },
        x=pos_labels,
        y=layer_labels,
        color_continuous_scale='RdBu',
        aspect='auto',
        title='Patching Experiment: Logit Difference Heatmap'
    )
    fig.update_xaxes(tickangle=45, tickmode='array', tickvals=list(range(seq_len)), ticktext=pos_labels)
    fig.update_traces(
        hovertemplate='Layer: %{y}<br>Token Position: %{x}<br>Logit Diff: %{z:.4f}'
    )
    fig.show()

# =====================================================================================
# Patches the target hook with source hook, during generation. If token_position=None, patches at all token positions. 
#Note, source hook should occur before the target hook
def simple_activation_patching(
    model: HookedTransformer,
    prompt: str,
    target_hook: str,
    source_hook: str,
    token_position: int,
    max_new_tokens: int = 20
):
    
    initial_tokens = model.to_tokens(prompt, prepend_bos=False)
    current_source_activations=None

    # Hook to capture source activations during generation
    def capture_source_hook(activations, hook):
        nonlocal current_source_activations
        current_source_activations = activations.clone()
        return activations
    
    # Hook to patch target activations with current source activations
    def patch_target_hook(activations, hook):
        if current_source_activations is None:
            return activations
            
        patched_activations = activations.clone()
        if token_position is None:
            # Patch all positions
            min_len = min(activations.shape[1], current_source_activations.shape[1])
            patched_activations[:, :min_len, :] = current_source_activations[:, :min_len, :]
        else:
            # Patch specific position
            if token_position < activations.shape[1] and token_position < current_source_activations.shape[1]:
                patched_activations[:, token_position, :] = current_source_activations[:, token_position, :]
        return patched_activations

    model.add_hook(source_hook, capture_source_hook)
    model.add_hook(target_hook, patch_target_hook)
    
    try:
        # Generate response with caching
        with torch.no_grad():
            current_tokens = initial_tokens.clone()
            new_str_tokens = []
            
            for step in range(max_new_tokens):
                
                # Run forward pass
                logits = model(current_tokens, prepend_bos=False)
                
                # Get next token (greedy decoding)
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                next_token_str = model.to_str_tokens(next_token)[0]
                new_str_tokens.append(next_token_str)
                
                # Early stopping
                if next_token_str == model.tokenizer.eos_token:
                    break
                    
                # Append new token
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)
                
                # Clear memory
                del logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Run final forward pass with cache
            final_logits, cache = model.run_with_cache(
                current_tokens,
                device=model.cfg.device,
                prepend_bos=False
            )
        
    finally:
        model.reset_hooks()# Clean up hook
    
    return {
            'activations': {key: value.cpu() for key, value in cache.items()},
            'tokens': current_tokens.cpu(),
            'str_tokens': model.to_str_tokens(current_tokens[0]),
            'input_str_tokens': initial_tokens,
            'generated_str_tokens': new_str_tokens,
            'seq_len': current_tokens.shape[1],
            'input_text': prompt
        }
