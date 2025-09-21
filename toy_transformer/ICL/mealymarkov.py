"""
Mealy-style Markov model generator.

Implements the model described by the user:
- n states (basis = one-hot rows)
- vocabulary size V
- a transition matrix T^i for every token i (each T^i is n x n)
- requires that T = sum_i T^i is a stochastic matrix (each row sums to 1)

Update rule used:
  - Given current state distribution eta^t (row vector, shape (n,)), the unnormalized
    probability mass of choosing token i is p_i_unnorm = eta^t @ T^i @ ones.
  - Normalize p = p / p.sum() to obtain distribution over tokens.
  - Sample token k ~ p.
  - Evolve the state:
        eta^{t+1} = (eta^t @ T^k) / (eta^t @ T^k @ ones)

The module validates dimensions and the stochasticity constraint.
It also logs the sequence of emitted tokens and the sequence of state distributions.

Example usage is included at the bottom under `if __name__ == '__main__':`.
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.special import rel_entr


class MarkovMealyModel:
    """Mealy Markov model where each token has an associated n x n transition matrix T^i.

    Attributes
    ----------
    n: int
        Number of states.
    V: int
        Vocabulary size (number of distinct tokens).
    T_list: List[np.ndarray]
        List of length V where each element is an (n x n) numpy array T^i.
    eta0: np.ndarray
        Initial state distribution (row vector shape (n,)). Defaults to uniform [1/n,...].
    basis: np.ndarray
        Identity matrix rows (one-hot basis states) shape (n, n).
    rng: np.random.Generator
        Random generator used for sampling.

    Note: 
        The states are implicitly numbered depending how the transition matrices are defined.
        This puts the burden of maintaining consistency across generations on the user.
    """

    def __init__(self, n: int, V: int, T_list: List[np.ndarray], eta0: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None):
        self.n = int(n)
        self.V = int(V)
        if len(T_list) != V:
            raise ValueError(f"Expected {V} transition matrices, got {len(T_list)}")

        # Convert to arrays and validate shapes
        self.T_list = []
        for i, T in enumerate(T_list):
            A = np.array(T, dtype=float)
            if A.shape != (self.n, self.n):
                raise ValueError(f"T^{i} has shape {A.shape}, expected ({self.n},{self.n})")
            if np.any(A < -1e-12):
                raise ValueError(f"T^{i} contains negative entries")
            # Small negative tolerance for floating rounding
            A[A < 0] = 0.0
            self.T_list.append(A)

        # Stochasticity check: sum_i T^i must be row-stochastic (rows sum to 1)
        T_sum = sum(self.T_list)
        row_sums = T_sum.sum(axis=1)
        if not np.allclose(row_sums, np.ones(self.n), atol=1e-8):
            raise ValueError("Sum of T^i across vocabulary is not row-stochastic. Row sums: " + str(row_sums))

        self.basis = np.eye(self.n)

        if eta0 is None:
            self.eta0 = np.full((self.n,), 1.0 / self.n)
        else:
            eta0 = np.array(eta0, dtype=float)
            if eta0.shape != (self.n,):
                raise ValueError(f"eta0 must be shape ({self.n},), got {eta0.shape}")
            if np.any(eta0 < -1e-12):
                raise ValueError("eta0 has negative entries")
            # normalize eta0
            s = eta0.sum()
            if s <= 0:
                raise ValueError("eta0 must sum to a positive value")
            self.eta0 = eta0 / s

        self.rng = rng if rng is not None else np.random.default_rng()

    def token_probabilities(self, eta: np.ndarray) -> np.ndarray:
        """Return probability distribution over tokens given current state distribution eta (row vector).

        p_i_unnorm = eta @ T^i @ ones
        p = p_i_unnorm / p_i_unnorm.sum()
        """
        eta = np.array(eta, dtype=float).reshape((self.n,))
        p_unnorm = np.empty((self.V,), dtype=float)
        ones = np.ones((self.n,))
        for i in range(self.V):
            rowvec = eta @ self.T_list[i]  # shape (n,)
            p_unnorm[i] = float(rowvec @ ones)  # scalar

        total = p_unnorm.sum()
        if total <= 0:
            raise RuntimeError("Total emission probability across tokens is zero. Model ill-defined for this eta.")
        #p = p_unnorm / total
        # TODO: Provide a proof that p_unnorm is already normalized.
        p = p_unnorm # I don't think they should be normalized once again. Well defined T^{i} should lead to normalized p automatically
        # numerical correction
        p = np.clip(p, 0.0, 1.0)
        #p = p / p.sum()\
        # sanity assert
        assert np.allclose(p.sum(), 1.0, atol=1e-8), "Token probabilities do not sum to 1"
        return p

    def evolve(self, eta: np.ndarray, token_index: int) -> np.ndarray:
        """Compute eta^{t+1} from eta^t and chosen token index using the given formula.

        eta_next = (eta @ T^{token}) / (eta @ T^{token} @ ones)
        """
        eta = np.array(eta, dtype=float).reshape((self.n,))
        T = self.T_list[token_index]
        numer = eta @ T  # shape (n,)
        denom = float(numer.sum())
        if denom <= 0:
            raise RuntimeError("Denominator for normalization is zero (no mass after applying T^{token}).")
        eta_next = numer / denom
        return eta_next

    def sample_sequence(self, max_new_tokens: int, initial_eta: Optional[np.ndarray] = None,
                        seed: Optional[int] = None) -> Tuple[List[int], List[np.ndarray]]:
        """Generate a token sequence and log the states traversed.

        Parameters
        ----------
        max_new_tokens: int
            Maximum number of tokens to generate (stops after this many tokens).
        initial_eta: Optional[np.ndarray]
            If provided, used as eta^0 (must be length n). Otherwise the model's eta0 is used.
        seed: Optional[int]
            Seed for RNG to make sampling reproducible for this call.

        Returns
        -------
        tokens: List[int]
            Emitted token indices.
        states: List[np.ndarray]
            List of eta^t row vectors (length = len(tokens)+1), starting with eta^0 and including eta^{final}.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        eta = self.eta0 if initial_eta is None else np.array(initial_eta, dtype=float).reshape((self.n,))
        # normalize if needed
        s = eta.sum()
        if s <= 0:
            raise ValueError("initial eta must sum to positive value")
        eta = eta / s

        tokens: List[int] = []
        states: List[np.ndarray] = [eta.copy()]

        for _ in range(int(max_new_tokens)):
            p = self.token_probabilities(eta)
            token = rng.choice(self.V, p=p)
            tokens.append(int(token))

            eta = self.evolve(eta, token)
            states.append(eta.copy())

        return tokens, states
    def verify_sequence(self, t_sequence: List[str], p_sequence: List[List[float]], tolerance: float = 1e-6) -> Tuple[bool, int]:
        """
        Given a sequence, verifies whether the given markov model is capable of generating it.
        Checks whether the KL divergence of the sequence converges such that for all n > N, 
        the kl_div(p_markovmodel, p_transformer) < tol.

        Parameters:
        -------------
        t_sequence: List[str]
            List of strings giving the sequence of tokens from the transformer. Includes all tokens of the sequence,
            not just the generated ones. 
        p_sequence: List[List[float]]
            A nested list of dimension sequence lenght x vocab size.
            Provides the probability distribution over the tokens at each position of the sequence as predicted by the transformer
        tolerance: float
            Defines the acceptable tolerance for claiming that the sequence from the transformer is correct
            kl_div(p_markovmodel, p_transformer) < tol for all n > N; N < n < len(sequence)
        
        Returns:
        ------------
        is_converged: bool
        convergence_position: int
            The last position where kl_div(p_markovmodel, p_transformer) < tol
        """

        if len(t_sequence) != len(p_sequence):
            raise ValueError("t_sequence and p_sequence must be of the same length")
        
        # Convert token strings to indices (assuming tokens are already indices 0 to V-1)
        try:
            token_indices = [int(token) for token in t_sequence]
        except ValueError:
            raise ValueError("t_sequence must contain string representations of integers from 0 to V-1")
        
        # Validate token indices are within vocabulary size
        if any(idx < 0 or idx >= self.V for idx in token_indices):
            raise ValueError(f"Token indices must be between 0 and {self.V-1}")
        
        # Initialize markov model state
        eta = self.eta0.copy()
        
        # Store KL divergences for each position
        kl_divergences = []
        
        # Process each token in the sequence
        for i, (token_idx, transformer_probs) in enumerate(zip(token_indices, p_sequence)):
            # Get markov model's token probabilities for current state
            markov_probs = self.token_probabilities(eta)            
            kl_div_value = rel_entr(np.array(markov_probs), np.array(transformer_probs)+1e-8).sum().item()
            kl_divergences.append(kl_div_value)
            
            # Log the KL divergence
            print(f"Position {i}: KL divergence = {kl_div_value:.6f}")
            print(f"\t\t transformer_probs = {transformer_probs}, Markov_probs = {markov_probs}")
            
            # Evolve the markov model state using the transformer's predicted token
            eta = self.evolve(eta, token_idx)
        
        # Check for convergence: find if there exists N such that for all n > N, kl_div < tolerance
        is_converged = False
        convergence_position = -1
        
        # Find the last position where KL divergence is above tolerance
        for i in range(len(kl_divergences) - 1, -1, -1):
            if kl_divergences[i] >= tolerance:
                convergence_position = i
                break
        
        # Check if all positions after convergence_position have KL divergence < tolerance
        if convergence_position == -1:
            # All positions have KL divergence < tolerance
            is_converged = True
            convergence_position = 0
        elif convergence_position < len(kl_divergences) - 1:
            # Check if all positions after convergence_position have KL divergence < tolerance
            remaining_kl_divs = kl_divergences[convergence_position + 1:]
            is_converged = all(kl < tolerance for kl in remaining_kl_divs)
        else:
            # Last position is above tolerance
            is_converged = False
        
        print(f"Convergence analysis:")
        print(f"  Convergence position: {convergence_position}")
        print(f"  Is converged: {is_converged}")
        print(f"  Final KL divergence: {kl_divergences[-1]:.6f}")
        
        return is_converged, convergence_position


if __name__ == '__main__':
    # Example small model (n=4 states, V=2 tokens) that satisfies the constraints.
    n = 3
    V = 2

    # We construct T^0 and T^1 so that T^0 + T^1 is row-stochastic (rows sum to 1).
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

    model = MarkovMealyModel(n=n, V=V, T_list=[T0, T1])

    # By specification the default eta^0 is uniform
    print("Initial eta^0 =", model.eta0)

    # Generate 12 tokens
    tokens, states = model.sample_sequence(max_new_tokens=30, seed=42)

    print("Generated tokens:", tokens)
    print("States (eta^t) traversed:")
    for i, s in enumerate(states):
        print(f"t={i} ->", np.round(s, 4))

    # You can also start from a basis state (one-hot). For example start at state 0:
    eta_onehot = model.basis[0]
    tokens2, states2 = model.sample_sequence(max_new_tokens=8, initial_eta=eta_onehot, seed=123)
    print('\nStarting from basis state 0:')
    print('tokens:', tokens2)
    print('states:')
    for i, s in enumerate(states2):
        print(f"t={i} ->", np.round(s, 4))
