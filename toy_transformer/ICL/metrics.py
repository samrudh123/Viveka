
from operator import ge
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import wandb
from itertools import product
from mealymarkov import MarkovMealyModel
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name



def get_ngram_stats(x: torch.Tensor, n: int) -> Tuple[Dict[str, float], Dict[str, int], int]:
    """
    Compute n-gram frequencies from sequence data.

    Args:
        x: Tensor of shape (batch_size, sequence_length) containing integer sequences
        n: N-gram size (2 for bigram, 3 for trigram, etc.)

    Returns:
        Tuple of (frequencies_dict, counts_dict, total_ngrams)
    """
    if n < 1:
        raise ValueError("N-gram size must be at least 1")

    # Generate all possible n-grams for the vocabulary
    vocab_size = x.max().item() + 1
    ngram_counts = {}

    # Initialize all possible n-grams to 0
    for pattern in product(range(vocab_size), repeat=n):
        pattern_str = ''.join(map(str, pattern))
        ngram_counts[pattern_str] = 0

    total_ngrams = 0

    # Count n-grams in the data
    for i in range(x.shape[0]):  # for each sequence
        for j in range(x.shape[1] - n + 1):  # for each position where n-gram fits
            ngram = ''.join([str(x[i, j+k].item()) for k in range(n)])
            if ngram in ngram_counts:
                ngram_counts[ngram] += 1
                total_ngrams += 1

    # Convert to frequencies
    ngram_freqs = {}
    for pattern, count in ngram_counts.items():
        freq = count / total_ngrams if total_ngrams > 0 else 0
        ngram_freqs[pattern] = freq

    return ngram_freqs, ngram_counts, total_ngrams


def ngram_kl(model, data:torch.Tensor, n: int, T_matrices: Optional[List[np.ndarray]] = None) -> float:
    """
    Compute KL divergence between true n-gram-based Markov process and model predictions.
    """
    # Default transition matrices if not provided
    if T_matrices is None:
        T0 = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0.5]
        ])
        T1 = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0.5, 0, 0]
        ])
        T_matrices = [T0, T1]

    # Import MarkovData here to avoid circular imports
    from toy_model import MarkovData

    # Generate test data
    if data is None:
        test_data = MarkovData(100, 32, 3, 2, T_matrices, seed=42)
        x = torch.stack(test_data.data)  # shape: (50, 32)
    else:
        x=data

    batch_size, seq_len = x.shape
    dist1 = torch.zeros(batch_size, seq_len, 2)  # shape: (50, 32, 2)

    if n == 1:  
        dist1[..., 0] = 0.5  # P(0)
        dist1[..., 1] = 0.5  # P(1)

    elif n == 2: 
        dist1[..., 0] = torch.where(x == 1, 1, 0.5)  # P(0)
        dist1[..., 1] = torch.where(x == 1, 0, 0.5)   # P(1)

    elif n==3:  
        for i in range(batch_size):
            for j in range(n-1, seq_len):
                # Get previous (n-1) tokens as context
                context = ''.join([str(x[i, j-k].item()) for k in range(n-1, 0, -1)])

                if context == "00":
                    dist1[i, j, 0] = 0.5  # P(0|00)
                    dist1[i, j, 1] = 0.5  # P(1|00)
                elif context == "01":
                    dist1[i, j, 0] = 1.0  # P(0|01)
                    dist1[i, j, 1] = 0.0  # P(1|01)
                elif context == "10":
                    dist1[i, j, 0] = 1.0  # P(0|10)
                    dist1[i, j, 1] = 0.0  # P(1|10)
                elif context == "11":  # This should never occur in the true process
                    dist1[i, j, 0] = 0.5  # Fallback
                    dist1[i, j, 1] = 0.5
    else:
        # For higher-order n-grams, use uniform distribution as fallback
        dist1[i, j, 0] = 0.5
        dist1[i, j, 1] = 0.5

    start_pos = n - 1

    # Get predicted probabilities from model
    dist2 = model(x).softmax(dim=-1)  # shape: (50, 32, 2)

    # Only evaluate positions where we have proper context
    dist1_eval = dist1[:, start_pos:]
    dist2_eval = dist2[:, start_pos:]

    # Avoid log(0) by clamping very small values
    eps = 1e-8
    dist1_clamped = dist1_eval.clamp(min=eps)
    dist2_clamped = dist2_eval.clamp(min=eps)

    # Compute KL divergence manually: sum_i P_i * log(P_i/Q_i)
    def kl_div(dist1_clamped, dist2_clamped):
        kl_div = dist1_clamped * (dist1_clamped.log() - dist2_clamped.log())
        kl_sum = kl_div.sum(dim=-1)  # sum over distribution axis
        return kl_sum.mean().item()  # scalar

    return kl_div(dist1_clamped, dist2_clamped), kl_div(dist2_clamped, dist1_clamped)

def markov_kl_proc(model, markov_data=None, process_id: int = 0) -> Tuple[float, float]:
    """
    Compute KL divergence between actual Markov process states and model predictions.
    This uses the true Markov state probabilities computed from the process itself.

    Args:
        model: The transformer model
        markov_data: MarkovData object. If None, creates default process.
        process_id: ID for labeling different processes in WandB

    Returns:
        Tuple of (markov_to_model_kl, model_to_markov_kl)
    """
    # Create default Markov process if none provided
    if markov_data is None:
        from toy_model import MarkovData
        T0 = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0.5]
        ])
        T1 = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0.5, 0, 0]
        ])

        markov_data = MarkovData(n_gen=50, gen_len=30, n_states=3, d_vocab=2, T_list=[T0, T1], seed=42 + process_id)

    # Get sequences and states
    x = torch.stack(markov_data.data)  # shape: (n_gen, gen_len)

    # Compute true probabilities from Markov states
    dist1 = []
    for etas in markov_data.states:
        sequence_probs = []
        for eta in etas[1:]:  # Skip first state, start from position 1
            token_probs = markov_data.model.token_probabilities(eta)
            sequence_probs.append(token_probs)
        dist1.append(sequence_probs)

    dist1 = torch.tensor(np.array(dist1), dtype=torch.float32)  # shape: (n_gen, gen_len-1, d_vocab)

    # Get predicted probabilities from model (skip first token for prediction)
    dist2 = model(x).softmax(dim=-1)[:, :-1, :]  # shape: (n_gen, gen_len-1, d_vocab)

    # Ensure same shape
    min_len = min(dist1.shape[1], dist2.shape[1])
    dist1 = dist1[:, :min_len, :]
    dist2 = dist2[:, :min_len, :]

    # Avoid log(0) by clamping very small values
    eps = 1e-8
    dist1_clamped = dist1.clamp(min=eps)
    dist2_clamped = dist2.clamp(min=eps)

    # Compute KL divergence manually: sum_i P(i) * log(P(i)/Q(i))
    def kl_divergence(p, q):
        kl_div = p * (p.log() - q.log())
        kl_sum = kl_div.sum(dim=-1)  # sum over vocabulary
        return kl_sum.mean().item()  # scalar

    markov_to_model = kl_divergence(dist1_clamped, dist2_clamped)
    model_to_markov = kl_divergence(dist2_clamped, dist1_clamped)

    return markov_to_model, model_to_markov


def compute_composition_scores(model, layer1_idx: int = 0, layer2_idx: int = 1) -> Dict[str, float]:
    """
    Compute Q, K, V composition scores between attention heads in different layers.
    Based on Elhage et al. [9]: ||MW^h_OV||_F / (||M||_F ||W^h_OV||_F)

    Args:
        model: HookedTransformer model
        layer1_idx: First layer index  
        layer2_idx: Second layer index (should be > layer1_idx)

    Returns:
        Dict with composition scores for each head pair
    """
    composition_scores = {}

    if layer2_idx >= len(model.blocks) or layer1_idx >= layer2_idx:
        return composition_scores

    layer1_attn = model.blocks[layer1_idx].attn
    layer2_attn = model.blocks[layer2_idx].attn

    n_heads = layer1_attn.cfg.n_heads

    for h1 in range(n_heads):
        for h2 in range(n_heads):
            try:
                # Get weight matrices
                W1_V = layer1_attn.W_V[h1]  # [d_model, d_head]
                W1_O = layer1_attn.W_O[h1]  # [d_head, d_model] 
                W1_OV = W1_V @ W1_O         # [d_model, d_model]

                W2_Q = layer2_attn.W_Q[h2]  # [d_model, d_head]
                W2_K = layer2_attn.W_K[h2]  # [d_model, d_head]
                W2_V = layer2_attn.W_V[h2]  # [d_model, d_head]

                # Compute composition matrices
                M_QK = W2_Q.T @ W1_OV       # Q-composition
                M_KK = W2_K.T @ W1_OV       # K-composition  
                M_VK = W2_V.T @ W1_OV       # V-composition

                # Compute Frobenius norms
                def frobenius_norm(x):
                    return torch.norm(x, p='fro').item()

                W1_OV_norm = frobenius_norm(W1_OV)

                # Q-composition score
                if W1_OV_norm > 1e-8:
                    q_score = frobenius_norm(M_QK) / (frobenius_norm(W2_Q.T) * W1_OV_norm)
                    composition_scores[f'q_comp_{layer1_idx}_{h1}_to_{layer2_idx}_{h2}'] = q_score
                    k_score = frobenius_norm(M_KK) / (frobenius_norm(W2_K.T) * W1_OV_norm)
                    composition_scores[f'k_comp_{layer1_idx}_{h1}_to_{layer2_idx}_{h2}'] = k_score
                    v_score = frobenius_norm(M_VK) / (frobenius_norm(W2_V.T) * W1_OV_norm)
                    composition_scores[f'v_comp_{layer1_idx}_{h1}_to_{layer2_idx}_{h2}'] = v_score

            except Exception as e:
                print(f"Error computing composition for heads {h1}-{h2}: {e}")
                continue

    return composition_scores


def compute_previous_token_matching_score(model, data: torch.Tensor) -> Dict[str, float]:
    """
    Compute previous-token matching scores for all attention heads.
    Measures how much each attention head attends to the immediately preceding token.
    """
    device = next(model.parameters()).device
    scores = {}
    sequences = []
    if data is None:
        data = torch.randint(0, model.cfg.d_vocab, (100, 32))
    
    sequences = data[:100].to(device)  # [num_samples, seq_len]
    seq_len = sequences.shape[1]


    with torch.no_grad():
        # Run model and get attention patterns
        hook_names = []
        for layer_idx in range(model.cfg.n_layers):
            # Use get_act_name for proper hook naming
            hook_names.append(get_act_name("pattern", layer_idx))

        # Run model with specific hooks cached
        logits, cache = model.run_with_cache(sequences, names_filter=hook_names)


        for layer_idx in range(model.cfg.n_layers):
            pattern_hook_name = get_act_name("pattern", layer_idx)

            if pattern_hook_name in cache:
                attn_patterns = cache[pattern_hook_name]  # [batch, head, seq_len, seq_len]

                # Validate shape
                if len(attn_patterns.shape) != 4:
                    print(f"Warning: Unexpected attention pattern shape for layer {layer_idx}: {attn_patterns.shape}")
                    continue

                batch_size, n_heads, attn_seq_len, _ = attn_patterns.shape
            for head_idx in range(model.cfg.n_heads):
                # Get attention from each token to the previous token
                # attn_patterns[:, head_idx, i, i-1] gives attention from token i to token i-1
                prev_token_scores = []

                for seq_pos in range(1, seq_len):  # Start from position 1
                    # Attention from position seq_pos to position seq_pos-1
                    attention_to_prev = attn_patterns[:, head_idx, seq_pos, seq_pos-1]
                    prev_token_scores.extend(attention_to_prev.cpu().numpy())

                # Average score across all positions and sequences
                avg_score = np.mean(prev_token_scores) if prev_token_scores else 0.0
                scores[f'prev_token_l{layer_idx}_h{head_idx}'] = avg_score

    return scores


def compute_in_context_learning_score(model, data:Optional[torch.Tensor], k1: int = 5, k2: int = 32) -> float:
    """
    Compute in-context learning score: ICL_{k1,k2}(w) = ℓ_{n,k1}(w) - ℓ_{n,k2}(w)
    Measures relative performance later in sequence vs earlier.
    """
    device = next(model.parameters()).device

    if data is None:
        #max_len=min(model.cfg.n_ctx, 1024)
        #data = torch.randint(0, model.cfg.d_vocab, (100, max_len), device=device)
        max_len = min(model.cfg.n_ctx, 32)
        T0 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0.5]])
        T1 = np.array([[0, 0, 0], [0, 0, 0], [0.5, 0, 0]])
        from toy_model import MarkovData
        markov_data = MarkovData(n_gen=100, gen_len=max_len, n_states=3, d_vocab=2, T_list=[T0, T1], seed=43)
        data = torch.stack(markov_data.data)
    num_samples=min(100, data.shape[0])
    sequences=data[:num_samples].to(device) 

    seq_len=sequences.shape[1]
    k1 = min(k1, seq_len - 2)  # Ensure k1 is within bounds
    k2 = min(k2, seq_len - 2)  # Ensure k2 is within bounds
    if k1 >= k2:
        raise ValueError("k1 must be less than k2")

    with torch.no_grad():
        logits = model(sequences)

        # Compute losses at positions k1 and k2
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        # Loss at position k1 (predicting token at k1+1)
        if k1 < seq_len - 1:
            logits_k1 = logits[:, k1]  # [batch, vocab]
            targets_k1 = sequences[:, k1 + 1]  # [batch]
            losses_k1 = loss_fn(logits_k1, targets_k1)  # [batch]
            avg_loss_k1 = losses_k1.mean().item()
        else:
            avg_loss_k1 = float('inf')

        # Loss at position k2 (predicting token at k2+1) 
        if k2 < seq_len - 1:
            logits_k2 = logits[:, k2]  # [batch, vocab]
            targets_k2 = sequences[:, k2 + 1]  # [batch]
            losses_k2 = loss_fn(logits_k2, targets_k2)  # [batch]
            avg_loss_k2 = losses_k2.mean().item()
        else:
            avg_loss_k2 = float('inf')

        # ICL score: loss at early position - loss at later position
        # More negative = better in-context learning
        icl_score = avg_loss_k1 - avg_loss_k2

    return icl_score

def generate_prefix_matching_data(n_samples: int = 100, seq_len: int = 64, 
                                vocab_size: int = 2, seed: int = 42) -> torch.Tensor:
    """Generate data with repeated tokens for prefix matching analysis."""
    torch.manual_seed(seed)
    sequences = []

    for _ in range(n_samples):
        seq = torch.randint(0, vocab_size, (seq_len,))

        # Insert repeated tokens: place token A in first half, then again in second half
        if seq_len >= 4:
            token_A = torch.randint(0, vocab_size, (1,)).item()
            first_half_end = max(2, seq_len // 3)
            second_half_start = max(first_half_end + 5, 2 * seq_len // 3)

            if first_half_end < second_half_start < seq_len:
                first_pos = torch.randint(1, first_half_end, (1,)).item()
                second_pos = torch.randint(second_half_start, seq_len, (1,)).item()

                seq[first_pos] = token_A
                seq[second_pos] = token_A

        sequences.append(seq)

    return torch.stack(sequences)


def compute_prefix_matching_score(model, data:Optional[torch.Tensor]=None) -> Dict[str, float]:
    """
    Compute prefix matching scores for all attention heads.
    Measures how much attention heads attend back to first instance of token 
    when encountering second instance.
    """
    device = next(model.parameters()).device
    scores = {}
    # Generate sequences with repeated tokens
    sequences = []
    if data is None:
        vocab_size = model.cfg.d_vocab
        data = generate_prefix_matching_data(n_samples=100, seq_len=64, vocab_size=vocab_size)

    sequences=data.to(device)
    num_samples, seq_len=sequences.shape

    with torch.no_grad():
        _, cache = model.run_with_cache(sequences)

        for layer_idx in range(model.cfg.n_layers):
            cache_key = f'blocks.{layer_idx}.attn.pattern'
            if cache_key not in cache:
                continue
            attn_patterns = cache[cache_key]  # [batch, head, seq_len, seq_len]
            for head_idx in range(model.cfg.n_heads):
                prefix_scores = []

                for batch_idx in range(num_samples):
                    # Find positions of repeated tokens
                    seq = sequences[batch_idx]

                    for second_pos in range(seq_len // 2 + 1, seq_len):
                        token_at_second = seq[second_pos].item()

                        # Find first occurrence of same token
                        first_positions = []
                        for first_pos in range(second_pos):
                            if seq[first_pos].item() == token_at_second:
                                first_positions.append(first_pos)

                        if first_positions:
                            # Get attention from second_pos to first occurrence
                            first_pos = first_positions[0]  # Take earliest occurrence
                            attention_score = attn_patterns[batch_idx, head_idx, second_pos, first_pos].item()
                            prefix_scores.append(attention_score)

                avg_score = np.mean(prefix_scores) if prefix_scores else 0.0
                scores[f'prefix_match_l{layer_idx}_h{head_idx}'] = avg_score

    return scores

class MetricsConfig:
    """Configuration for metrics tracking."""
    def __init__(
        self,
        # N-gram metrics
        track_ngrams: bool = True,
        ngram_orders: List[int] = [1, 2, 3],
        ngram_data: Optional[torch.Tensor] = None,

        # Markov KL metrics
        track_markov_kl: bool = True,
        markov_processes: List = None,  # List of MarkovData objects or None for defaults

        # Composition metrics
        track_composition: bool = True,
        composition_layers: List[Tuple[int, int]] = [(0, 1)],

        # Previous token matching
        track_previous_token: bool = True,
        prev_token_data: Optional[torch.Tensor] = None,

        # In-context learning
        track_in_context: bool = True,
        icl_data: Optional[torch.Tensor] = None,
        icl_k1: int = 10,
        icl_k2: int = 50,

        # Prefix matching
        track_prefix_matching: bool = False,
        prefix_data: Optional[torch.Tensor] = None,
    ):
        self.track_ngrams = track_ngrams
        self.ngram_orders = ngram_orders
        self.ngram_data = ngram_data

        self.track_markov_kl = track_markov_kl
        self.markov_processes = markov_processes or []

        self.track_composition = track_composition
        self.composition_layers = composition_layers

        self.track_previous_token = track_previous_token
        self.prev_token_data = prev_token_data

        self.track_in_context = track_in_context
        self.icl_data = icl_data
        self.icl_k1 = icl_k1
        self.icl_k2 = icl_k2

        self.track_prefix_matching = track_prefix_matching
        self.prefix_data = prefix_data


def compute_metrics(model, config: MetricsConfig, step: int) -> Dict[str, float]:
    """
    Compute individual metrics based on configuration.
    Each metric uses its own optimal data if not provided.
    """
    metrics_to_log = {}

    try:
        # 1. N-gram KL divergence metrics
        if config.track_ngrams:
            for n in config.ngram_orders:
                try:
                    kl_true_to_model, kl_model_to_true = ngram_kl(model, config.ngram_data, n)
                    metrics_to_log[f'{n}gram_kl_true_to_model'] = kl_true_to_model
                    metrics_to_log[f'{n}gram_kl_model_to_true'] = kl_model_to_true 
                except Exception as e:
                    print(f"Error computing {n}-gram KL: {e}")

        # 2. Composition scores  
        if config.track_composition:
            try:
                for layer1_idx, layer2_idx in config.composition_layers:
                    comp_scores = compute_composition_scores(model, layer1_idx, layer2_idx)

                    # Log individual composition scores for visualization
                    for k, v in comp_scores.items():
                        metrics_to_log[k] = v

                    # Also log averages
                    if comp_scores:
                        q_scores = [v for k, v in comp_scores.items() if k.startswith('q_comp')]
                        k_scores = [v for k, v in comp_scores.items() if k.startswith('k_comp')]
                        v_scores = [v for k, v in comp_scores.items() if k.startswith('v_comp')]

                        if q_scores: metrics_to_log[f'avg_q_composition_l{layer1_idx}_l{layer2_idx}'] = np.mean(q_scores)
                        if k_scores: metrics_to_log[f'avg_k_composition_l{layer1_idx}_l{layer2_idx}'] = np.mean(k_scores)
                        if v_scores: metrics_to_log[f'avg_v_composition_l{layer1_idx}_l{layer2_idx}'] = np.mean(v_scores)

            except Exception as e:
                print(f"Error computing composition scores: {e}")

        # 3. Previous token matching scores
        if config.track_previous_token:
            try:
                prev_scores = compute_previous_token_matching_score(model, config.prev_token_data)

                # Log individual head scores
                for k, v in prev_scores.items():
                    metrics_to_log[k] = v

                # Log averages per layer and overall
                if prev_scores:
                    all_scores = list(prev_scores.values())
                    metrics_to_log['avg_prev_token_matching'] = np.mean(all_scores)

                    for layer_idx in range(model.cfg.n_layers):
                        layer_scores = [v for k, v in prev_scores.items() if f'_l{layer_idx}_' in k]
                        if layer_scores:
                            metrics_to_log[f'prev_token_l{layer_idx}'] = np.mean(layer_scores)

            except Exception as e:
                print(f"Error computing previous token scores: {e}")

        # 4. In-context learning score
        if config.track_in_context:
            try:
                icl_score = compute_in_context_learning_score(
                    model, config.icl_data, config.icl_k1, config.icl_k2
                )
                metrics_to_log['in_context_learning'] = icl_score
            except Exception as e:
                print(f"Error computing in-context learning score: {e}")

        # 5. Prefix matching scores
        if config.track_prefix_matching:
            try:
                prefix_scores = compute_prefix_matching_score(model, config.prefix_data)

                # Log individual scores
                for k, v in prefix_scores.items():
                    metrics_to_log[k] = v

                # Log averages
                if prefix_scores:
                    all_scores = list(prefix_scores.values())
                    metrics_to_log['avg_prefix_matching'] = np.mean(all_scores)

                    for layer_idx in range(model.cfg.n_layers):
                        layer_scores = [v for k, v in prefix_scores.items() if f'_l{layer_idx}_' in k]
                        if layer_scores:
                            metrics_to_log[f'prefix_match_l{layer_idx}'] = np.mean(layer_scores)

            except Exception as e:
                print(f"Error computing prefix matching scores: {e}")

        if config.track_markov_kl:
            try:
                for pid, markov_data in enumerate(config.markov_processes):
                    markov_to_model, model_to_markov = markov_kl_proc(model, markov_data, process_id=pid)
                    metrics_to_log[f'markov{pid}_to_model_kl'] = markov_to_model
                    metrics_to_log[f'model_to_markov{pid}_kl'] = model_to_markov
            except Exception as e:
                print(f"Error computing Markov KL metrics: {e}")        

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log(metrics_to_log, step=step)

    except Exception as e:
        print(f"Error in metrics computation at step {step}: {e}")

    return metrics_to_log

class CleanMetricsTracker:
 #backward compatibility wrapper
    def __init__(self, ngram_orders=[1, 2, 3], track_composition=True, 
                 track_previous_token=True, track_in_context=True, 
                 track_prefix_matching=False):
        self.config = MetricsConfig(
            track_ngrams=True,
            ngram_orders=ngram_orders,
            track_composition=track_composition,
            track_previous_token=track_previous_token,
            track_in_context=track_in_context,
            track_prefix_matching=track_prefix_matching
        )
        self.metrics = {}

    def compute_all_metrics(self, model, data: torch.Tensor, step: int):
        """Backward compatibility method."""
        # Use the provided data for all metrics (old behavior)
        self.config.ngram_data = data
        self.config.prev_token_data = data
        self.config.icl_data = data
        self.config.prefix_data = data

        metrics = compute_metrics(model, self.config, step)
        self.metrics[step] = metrics
        return metrics

    def get_metrics_history(self):
        return self.metrics

   
 
