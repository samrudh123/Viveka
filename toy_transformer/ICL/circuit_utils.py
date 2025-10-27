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
import re

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

    # Check if the model *has* a final layer norm
    if hasattr(model, 'ln_final') and model.ln_final is not None:
        normalized = model.ln_final(activations)
    else:
        # If no ln_final, just use the raw activations
        normalized = activations
        
    if norm:
        # This is the custom normalized unembedding from circuit_utils.py
        # We must use model.W_U (the unembedding matrix) here
        logits = normalized @ F.normalize(model.W_U, p=p, dim=0)
    else:
        # This is the standard unembedding
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
    try:
        # --- FIX ---
        # Correctly index the cache: [batch_dim, token_dim]
        # We assume batch_size is 1, so we use index 0.
        activations = cache_data['activations'][hook_name][0, token_pos]
        # -----------
    except IndexError:
        print(f"IndexError: Failed to get activations. Your sequence length is {cache_data['seq_len']}.")
        print(f"Cannot access token_pos={token_pos}. Please choose a value from 0 to {cache_data['seq_len']-1}.")
        return
    except KeyError:
        print(f"KeyError: Hook name '{hook_name}' not found in cache.")
        return

    logits = logit_lens(model, activations, norm, p)

    # Get top-k logits and their indices
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # We apply softmax only to the top-k logits for visualization,
    # as in the original function's apparent intent.
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    
    # Convert indices to token strings
    top_k_tokens = model.to_str_tokens(top_k_indices)

    # Create the bar plot
    fig = go.Figure(go.Bar(
        x=[f"'{token}'" for token in top_k_tokens],
        y=top_k_probs,
        text=top_k_probs.cpu().numpy().round(2),
        textposition='auto'
    ))
    fig.update_layout(
        title=f"Logit Lens: Top {k} Predictions from {hook_name} at Position {token_pos}",
        xaxis_title="Token",
        yaxis_title="Probability",
    )
    fig.show()

def create_interactive_widget(model: HookedTransformer, cache_data: Dict,
                              norm: bool = False, p: float = 2, k: int = 10):
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

    k_slider = widgets.IntSlider(value=k, min=1, max=20, step=1, description='Top-K:',
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
    token_range: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (10, 5)
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
    full_hook_name = f'blocks.{layer_num}.attn.{hook_name}'
    
    try:
        # Access the activation, assuming batch_size=1 (index [0])
        activations = cache_data['activations'][full_hook_name][0].cpu().numpy()
    except KeyError:
        full_hook_name = f'blocks.{layer_num}.{hook_name}'
        activations = cache_data['activations'][full_hook_name][0].cpu().numpy()
    except Exception as e:
        print(f"Error accessing activations: {e}")
        return

    # Set token range
    if token_range is None:
        start, end = 0, cache_data['seq_len']
    else:
        start, end = token_range
    
    str_tokens = cache_data['str_tokens'][start:end]
    plot_labels = [f"'{tok}' (Pos {i+start})" for i, tok in enumerate(str_tokens)]
    n_heads = model.cfg.n_heads
    
    if hook_name == 'hook_pattern':
        # activations shape is [n_heads, seq_len, seq_len]
        data = activations[:, start:end, start:end]
        title = f'Attention Patterns: Layer {layer_num}'
        labels = {"x": "Key", "y": "Query", "color": "Score"}
    
    elif hook_name == 'hook_z':
        # activations shape is [seq_len, n_heads, d_head]
        # We compute the L2 norm of the value vector from each head
        data = np.linalg.norm(activations[start:end, :, :], axis=-1)
        # data shape is [seq_len, n_heads], transpose to [n_heads, seq_len]
        data = data.T 
        title = f'Value Vector L2 Norm: Layer {layer_num}'
        labels = {"x": "Token Position", "y": "Head", "color": "L2 Norm"}
        # For this plot, x-axis should be tokens, y-axis is heads
        fig = px.imshow(
            data,
            x=plot_labels,
            y=[f"Head {i}" for i in range(n_heads)],
            labels=labels,
            title=title,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=figsize[1]*50, width=figsize[0]*50)
        fig.show()
        return # Return early as the plot is different

    elif hook_name == 'hook_attn_out':
         # activations shape is [seq_len, d_model]
         # This is the *output* of the layer, not per-head.
         # We compute the L2 norm of the output vector
        data = np.linalg.norm(activations[start:end, :], axis=-1)
        # data shape is [seq_len], add a dummy dim [1, seq_len]
        data = np.expand_dims(data, axis=0)
        title = f'Layer Output L2 Norm: Layer {layer_num}'
        labels = {"x": "Token Position", "y": "Layer", "color": "L2 Norm"}
        # For this plot, x-axis is tokens, y-axis is just one row
        fig = px.imshow(
            data,
            x=plot_labels,
            y=[f"L{layer_num} Out"],
            labels=labels,
            title=title,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=figsize[1]*50, width=figsize[0]*50)
        fig.show()
        return # Return early

    else:
        print(f"Unsupported hook name for attention display: {hook_name}")
        return

    # Plot faceted heatmap for hook_pattern
    fig = px.imshow(
        data,
        facet_col=0,
        facet_col_wrap=n_heads,
        labels=labels,
        title=title,
        x=plot_labels,
        y=plot_labels,
        color_continuous_scale='Viridis'
    )
    
    # Set titles for each subplot
    for i in range(n_heads):
        fig.layout.annotations[i]['text'] = f'Head {i}'
        
    fig.show()

def create_attention_widget(model: HookedTransformer, cache_data: Dict, figsize: Tuple[int, int] = (10, 5)):
    """
    Launches an interactive widget to visualize attention patterns by layer, token range, and hook.

    Parameters
    ----------
    model : HookedTransformer
        The transformer model.
    cache_data : dict
        Dictionary with cached activations and tokens.
    figsize : tuple of int, optional
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
        display_attention_patterns(model, cache_data, layer, hook_name, token_range, figsize=figsize)

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
    hook_names = [hook for hook in model.hook_dict.keys() if re.search(hook_filter, hook)]
    heatmap_data = []

    for hook_name in hook_names:
        # --- FIX ---
        # Correctly index the cache: [batch_dim, token_range, ...]
        activations = cache_data['activations'][hook_name][0, start:]
        # -----------
        
        logits = logit_lens(model, activations, norm=False) # [seq_len, d_vocab]
        top_tokens = logits.argmax(dim=-1) # [seq_len]
        
        # Add a dummy dimension for stacking (becomes [1, seq_len])
        heatmap_data.append(top_tokens.unsqueeze(0))

    # Stack to [num_hooks, 1, seq_len]
    heatmap_data = torch.stack(heatmap_data)
    
    # --- FIX ---
    # Squeeze out the dummy batch dimension to get [num_hooks, seq_len]
    heatmap_data = heatmap_data.squeeze(1).cpu().numpy()
    # -----------

    str_tokens = cache_data['str_tokens'][start:]
    
    plot_labels = [f"'{tok}' (Pos {i+start})" for i, tok in enumerate(str_tokens)]
    
    fig = px.imshow(
        heatmap_data,
        x=plot_labels,
        y=hook_names,
        labels={"x": "Token", "y": "Layer", "color": "Top Token ID"},
        title="Logit Lens: Top Token Prediction by Layer"
    )
    fig.update_layout(height=figsize[1]*50, width=figsize[0]*50)
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

    if isinstance(tokens, str):
        token_id = model.to_single_token(tokens)
    else:
        token_id = [model.to_single_token(t) for t in tokens]
        
    hook_names = [hook for hook in model.hook_dict.keys() if re.search(hook_filter, hook)]
    heatmap_data = []

    for hook_name in hook_names:
        # --- FIX ---
        # Correctly index the cache: [batch_dim, token_range, ...]
        activations = cache_data['activations'][hook_name][0, start:]
        # -----------
        
        logits = logit_lens(model, activations, norm=True) # [seq_len, d_vocab]
        
        # Get rank of the token
        ranks = (logits.argsort(dim=-1, descending=True) == token_id).nonzero()[:, 1]
        
        # Add a dummy dimension for stacking (becomes [1, seq_len])
        heatmap_data.append(ranks.unsqueeze(0))

    # Stack to [num_hooks, 1, seq_len]
    heatmap_data = torch.stack(heatmap_data)
    
    # --- FIX ---
    # Squeeze out the dummy batch dimension to get [num_hooks, seq_len]
    heatmap_data = heatmap_data.squeeze(1).cpu().numpy()
    # -----------
    
    str_tokens = cache_data['str_tokens'][start:]
    fig = px.imshow(
        heatmap_data,
        x=str_tokens,
        y=hook_names,
        labels={"x": "Token", "y": "Layer", "color": "Rank"},
        title=f"Logit Lens: Rank of Token(s) '{tokens}' by Layer",
        color_continuous_scale='Viridis_r' # Invert so 0 (top rank) is bright
    )
    fig.update_layout(height=figsize[1]*50, width=figsize[0]*50)
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
    token_id = model.to_single_token(token)
    
    # Get the unembedding vector for the target token
    W_U = model.W_U # [d_model, d_vocab]
    W_U_token = W_U[:, token_id] # [d_model]

    all_head_contributions = []
    
    for layer in range(model.cfg.n_layers):
        # --- FIX ---
        # Get hook_z, shape [1, seq_len, n_heads, d_head]
        z_vectors = cache_data['activations'][f'blocks.{layer}.attn.hook_z']
        
        # Get W_O, shape [n_heads, d_head, d_model]
        W_O = model.blocks[layer].attn.W_O
        
        # Calculate head outputs: z @ W_O
        # z_vectors[0]: [seq_len, n_heads, d_head]
        # einsum: [seq_len, n_heads, d_head] @ [n_heads, d_head, d_model] -> [seq_len, n_heads, d_model]
        head_outputs = torch.einsum(
            "sph,hdm->shm", 
            z_vectors[0].cpu(), # <-- Index [0] to get batch
            W_O.cpu()
        )
        # -----------
        
        # Project onto the token's unembedding direction
        # [seq_len, n_heads, d_model] @ [d_model] -> [seq_len, n_heads]
        contribution = torch.einsum(
            "shm,m->sh",
            head_outputs,
            W_U_token.cpu()
        )
        all_head_contributions.append(contribution)

    # all_head_contributions is a list of [seq_len, n_heads] tensors
    # Stack them: [n_layers, seq_len, n_heads]
    contribution_tensor = torch.stack(all_head_contributions).numpy()

    # Create the interactive widget
    @widgets.interact(
        layer=widgets.IntSlider(min=0, max=model.cfg.n_layers - 1, step=1, value=0),
        token_pos=widgets.IntSlider(min=0, max=cache_data['seq_len'] - 1, step=1, value=cache_data['seq_len'] - 1)
    )
    def plot(layer, token_pos):
        data = contribution_tensor[layer, token_pos, :] # [n_heads]
        fig = px.bar(
            x=[f"Head {i}" for i in range(model.cfg.n_heads)],
            y=data,
            labels={"x": "Head", "y": "Contribution"},
            title=f"Head Contribution to '{token}' at Layer {layer} (Pos {token_pos})"
        )
        fig.show()

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
        - < 1: for flattening ie. making attention pattern more uniform
        - > 1: for amplifying ie. making attention pattern more peaked
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
    
# CAUTION: for efficiency, automatic differentiation is being disabled globally in PyTorch
# Shouldn't matter unless you want to finetune
if __name__ == "__main__":
    torch.set_grad_enabled(False)
