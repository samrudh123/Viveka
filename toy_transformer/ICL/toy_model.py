
import os

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import wandb
from typing import Optional, Literal

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.train import HookedTransformerTrainConfig
from metrics import MetricsConfig, generate_prefix_matching_data, compute_metrics
from mealymarkov import MarkovMealyModel

class MarkovData(Dataset):

    def __init__(
        self,
        n_gen: int,
        gen_len: int,
        n_states: int,
        d_vocab: int,
        T_list: list[np.ndarray],
        eta0: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        seed: int = 42
    ):
        self.model = MarkovMealyModel(n_states, d_vocab, T_list, eta0, rng)
        self.d_vocab = self.model.V
        self.gen_len = gen_len
        self.data = []
        self.states = []
        rng = rng or np.random.default_rng(seed)

        for i in range(n_gen):
            tokens, states = self.model.sample_sequence(
                max_new_tokens=gen_len,
                seed=rng.integers(2**32)
            )

            self.data.append(torch.tensor(tokens, dtype=torch.int64))
            self.states.append(states)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"tokens": self.data[idx]}


class MergeMarkovDatasets(Dataset):
    """
    Merges two MarkovData objects and returns a new Dataset object.

    Mixing style is the manner in which the datasets should be merged:
    - "random": Generations from both the datasets are randomly mixed.
    - "alternate": The new dataset has generations alternating from both the datasets.
    - "stack": Generations of the second dataset are added after generations of the first dataset.

    Note that `mixing_style` may play an important role in training of the model.
    """

    def __init__(
        self, 
        dataset1: MarkovData, 
        dataset2: MarkovData, 
        mixing_style: Literal["random", "alternate", "stack"]
    ):
        self.model1 = dataset1.model
        self.model2 = dataset2.model

        assert dataset1.d_vocab == dataset2.d_vocab, "Vocabulary size for the datasets does not match"
        self.d_vocab = dataset1.d_vocab

        assert dataset1.gen_len == dataset2.gen_len, "Generation lengths for the datasets do not match"
        self.gen_len = dataset1.gen_len

        data1 = list(zip(dataset1.data, dataset1.states))
        data2 = list(zip(dataset2.data, dataset2.states))

        if mixing_style == "random":
            merged = data1 + data2
            np.random.shuffle(merged)
        elif mixing_style == "alternate":
            assert len(data1) == len(data2), "Mixing style 'alternate' is valid only when the size of both datasets is same"
            merged = []
            for i in range(len(data1)):
                merged.append(data1[i])
                merged.append(data2[i])
        else:
            merged = data1 + data2

        self.data = [d for d, s in merged]
        self.states = [s for d, s in merged]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"tokens": self.data[idx]}


def train(
    model: HookedTransformer,
    config: HookedTransformerTrainConfig,
    train_data: Dataset,
    val_data: Optional[Dataset] = None,
    eval_every: Optional[int] = None,
    metrics_config: Optional[MetricsConfig] = None,
    metrics_log_interval: int = 50
) -> HookedTransformer:
    """
    Helper function to train an HookedTransformer model on an autoregressive language modeling task.
    Slightly modified version of TransformerLens one with advanced metrics tracking integration.

    Args:
        model: The model to train
        config: The training configuration
        train_data: The dataset to train on
        val_data: The dataset to use for validation
        eval_every: Number of epochs after which to run the model on val_data
        metrics_tracker: AdvancedMetricsTracker instance for logging metrics

    Returns:
        The trained model
    """
    torch.manual_seed(config.seed)

    model.train()

    if config.wandb:
        if config.wandb_project_name is None:
            config.wandb_project_name = "easy-transformer"
        wandb.init(project=config.wandb_project_name, config=vars(config))

    # Set up optimizer
    if config.optimizer_name in ["Adam", "AdamW"]:
        if config.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=config.lr, 
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay if config.weight_decay is not None else 0.0,
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} not supported")

    # Set up learning rate scheduler
    scheduler = None
    if config.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, step / config.warmup_steps)
        )

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=len(val_data)) if val_data else None

    model = model.to(config.device)

    global_step = 0

    for epoch in tqdm(range(1, config.num_epochs + 1)):
        samples = 0
        for step, batch in enumerate(train_dataloader):
            tokens = batch["tokens"].to(config.device)

            loss = model(tokens, return_type="loss")
            loss.backward()

            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            if config.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()
            optimizer.zero_grad()

            samples += tokens.shape[0]
            global_step += 1

            if config.wandb:
                wandb.log({"train_loss": loss.item(), "samples": samples, "epoch": epoch}, step=global_step)

            if metrics_config is not None and global_step % metrics_log_interval == 0:
                #try:
                model.eval()
                with torch.no_grad():
                    compute_metrics(model, metrics_config, global_step)
                model.train()
                    
                if global_step % (metrics_log_interval) == 0:
                    print(f"Metrics logged at step {global_step}")
                #except Exception as e:
                 #   print(f"Warning: Error in metrics tracking at step {global_step}: {e}")
            

        if config.print_every is not None and epoch % config.print_every == 0:
            print(f"Epoch {epoch} Samples {samples} Step {step} Training Loss {loss.item()}")

        if config.save_every is not None and epoch % config.save_every == 0 and config.save_dir is not None:
            torch.save(model.state_dict(), f"{config.save_dir}/model{epoch}.pt")

        if val_dataloader and eval_every is not None and epoch % eval_every == 0:
            for data in val_dataloader:
                model.eval()
                tokens = data["tokens"].to(config.device)
                with torch.no_grad():
                    loss = model(tokens, return_type="loss")
                if config.wandb:
                    wandb.log({"val_loss": loss.item(), "epoch": epoch}, step=global_step)
                print(f"Epoch {epoch} Validation Loss {loss.item()}")

    return model


def train_model(
    dataset: MarkovData | MergeMarkovDatasets,
    # Transformer Architecture
    n_layers: int = 4,
    d_model: int = 64,
    n_heads: int = 1,
    d_head: int = 8,
    attn_only: bool = False,
    d_mlp: int = 256,
    act_fn: Literal["relu", "gelu", "silu", "gelu_new", "solu_ln", "gelu_fast"] = "relu",
    normalization_type: Literal["LN", "LNPre", "RMS", "RMSPre"] | None = None,
    positional_embedding_type: Literal["standard", "rotary", "shortformer"] = "standard",
    # Training Hyperparameters
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-2,
    optimizer_name: Literal["Adam", "AdamW", "SGD"] = "SGD",
    wandb: bool = False,
    wandb_project_name: Optional[str] = None,
    # System + I/O
    device: str = "cpu",
    seed: int = 42,
    save_every: int = 1,
    save_dir: str = "./checkpoints",
    print_every: int = 1,
    eval_every: int = 1,
    val_frac: float = 0.2,
    # Metrics tracking
    metrics_config: Optional[MetricsConfig] = None,
    metrics_log_interval: int = 50,

    track_ngrams: bool = True,
    ngram_orders: list[int] = [1,2, 3],
    track_previous_token: bool = True,
    track_in_context: bool = True,
    track_composition: bool = True,
    track_prefix_matching: bool = True
) -> HookedTransformer:
    """
    Train a HookedTransformer on sequences generated from a Mealy Markov model.

    This function constructs a HookedTransformer model with the given architecture 
    and optimization hyperparameters, and trains it on sequences generated from a 
    custom Markov process dataset.

    Parameters
    ----------
    dataset : MarkovData or MergeMarkovDatasets
        Training dataset containing token sequences generated from a Mealy-Markov process.
    n_layers : int
        Number of transformer layers.
    d_model : int
        Dimension of the model embedding and hidden sizes.
    n_heads : int
        Number of attention heads.
    d_head : int
        Dimension per attention head.
    attn_only : bool
        Whether the transformer is attention-only, without any MLP blocks.
    d_mlp : int
        Dimension of the feedforward hidden layer.
    act_fn : {"relu", "gelu", "silu", "gelu_new", "solu_ln", "gelu_fast"}
        Activation function used in MLP layers.
    normalization_type : {"LN", "LNPre", "RMS", "RMSPre"}
        Normalization strategy applied in transformer layers. Defaults to no normalization
    positional_embedding_type : {"standard", "rotary", "shortformer"}
        Type of positional embeddings used in the model
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate for optimization.
    optimizer_name : {"Adam", "AdamW", "SGD"}
        Optimizer to use.
    wandb : bool
        Whether to use wandb to log training
    wandb_project_name : str, optional
        Name for wandb project, defaults to "easy-transformer"
    device : str
        Device where the model will be trained (e.g., "cpu", "cuda").
    seed : int
        Random seed for reproducibility
    save_every : int
        Frequency (in epochs) to checkpoint the model.
    save_dir : str
        Directory where checkpoints will be saved.
    print_every : int
        Frequency (in epochs) to log training progress.
    eval_every : int
        Evaluate on a validation dataset
    val_frac : float
        Fraction of dataset to be used as validation dataset
    track_ngrams : bool
        Whether to track n-gram metrics during training
    ngram_orders : list[int]
        List of n-gram orders to track
    track_sets : list[str]
        List of dataset splits to track metrics for
    track_composition : bool
        Whether to track attention head composition scores
    track_previous_token : bool
        Whether to track previous token matching scores
    track_in_context : bool
        Whether to track in-context learning scores
    track_prefix_matching : bool
        Whether to track prefix matching scores

    Returns
    -------
    HookedTransformer
        The trained transformer model.
    """
    d_vocab = dataset.d_vocab
    n_ctx = dataset.gen_len

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_ctx=n_ctx,
        d_head=d_head,
        n_heads=n_heads,
        d_mlp=d_mlp,
        act_fn=act_fn,
        d_vocab=d_vocab,
        attn_only=attn_only,
        normalization_type=normalization_type,
        device=device,
        positional_embedding_type=positional_embedding_type,
        seed=seed,
        default_prepend_bos=False,
    )

    model = HookedTransformer(cfg, move_to_device=True)

    # System + I/O
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_dir is not None:
        torch.save(cfg, f"{save_dir}/model_cfg.pt")

    # System + I/O
    train_cfg = HookedTransformerTrainConfig(
        num_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer_name=optimizer_name,
        wandb=wandb,
        wandb_project_name=wandb_project_name,
        device=device,
        seed=seed,
        save_every=save_every,
        save_dir=save_dir,
        print_every=print_every
    )

    # Initialize ADVANCED metrics tracker if any metrics tracking is requested
    if metrics_config is None and any([track_ngrams, track_composition, track_previous_token, 
                                     track_in_context, track_prefix_matching]):
        metrics_config = MetricsConfig(
            track_ngrams=track_ngrams,
            ngram_orders=ngram_orders,
            track_composition=track_composition,
            track_previous_token=track_previous_token,
            track_in_context=track_in_context,
            track_prefix_matching=track_prefix_matching,
        )
        print("Created metrics config from individual parameters")

    # Train-val split
    if val_frac:
        train_size = int(len(dataset) * (1 - val_frac))
        indices = torch.randperm(len(dataset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_data = Subset(dataset, train_indices)
        val_data = Subset(dataset, val_indices)
        return train(model, train_cfg, train_data, val_data, eval_every, metrics_config)
    else:
        return train(model, train_cfg, dataset, metrics_config=metrics_config)


def finetune_model(
    model: HookedTransformer,
    dataset: MarkovData | MergeMarkovDatasets,
    n_epochs: int,
    batch_size: int = 64,
    lr: float = 1e-2,
    optimizer_name: Literal["Adam", "AdamW", "SGD"] = "SGD",
    wandb: bool = False,
    wandb_project_name: Optional[str] = None,
    device: str = "cpu",
    seed: int = 42,
    save_every: int = 1,
    save_dir: str = "./checkpoints",
    print_every: int = 1,
    eval_every: int = 1,
    val_frac: float = 0.2,
    # Advanced Metrics tracking
    track_ngrams: bool = True,
    ngram_orders: list[int] = [2, 3, 4],
    track_sets: list[str] = ["train", "val", "complete"],
    track_composition: bool = True,
    track_previous_token: bool = True,
    track_in_context: bool = True,
    track_prefix_matching: bool = True
) -> HookedTransformer:
    """
    Finetune a pretrained HookedTransformer on sequences generated from a Mealy Markov model

    Parameters
    ----------
    model : HookedTransformer
        A pre-trained model to finetune.
    dataset : MarkovData or MergeMarkovDatasets
        Training dataset containing token sequences generated from a Mealy-Markov process.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate for optimization.
    optimizer_name : {"Adam", "AdamW", "SGD"}
        Optimizer to use.
    wandb : bool
        Whether to use wandb to log training
    wandb_project_name : str, optional
        Name for wandb project, defaults to "easy-transformer"
    device : str
        Device where the model will be trained (e.g., "cpu", "cuda").
    seed : int
        Random seed for reproducibility
    save_every : int
        Frequency (in epochs) to checkpoint the model.
    save_dir : str
        Directory where checkpoints will be saved.
    print_every : int
        Frequency (in epochs) to log training progress.
    eval_every : int
        Evaluate on a validation dataset
    val_frac : float
        Fraction of dataset to be used as validation dataset
    track_ngrams : bool
        Whether to track n-gram metrics during training
    ngram_orders : list[int]
        List of n-gram orders to track
    track_sets : list[str]
        List of dataset splits to track metrics for
    track_composition : bool
        Whether to track attention head composition scores
    track_previous_token : bool
        Whether to track previous token matching scores
    track_in_context : bool
        Whether to track in-context learning scores
    track_prefix_matching : bool
        Whether to track prefix matching scores

    Returns
    -------
    HookedTransformer
        The trained transformer model.
    """
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cfg = HookedTransformerTrainConfig(
        num_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer_name=optimizer_name,
        wandb=wandb,
        wandb_project_name=wandb_project_name,
        device=device,
        seed=seed,
        save_every=save_every,
        save_dir=save_dir,
        print_every=print_every
    )
    # Train-val split
    if val_frac:
        train_size = int(len(dataset) * (1 - val_frac))
        indices = torch.randperm(len(dataset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_data = Subset(dataset, train_indices)
        val_data = Subset(dataset, val_indices)
        return train(model, cfg, train_data, val_data, eval_every, metrics_tracker)
    else:
        return train(model, cfg, dataset, metrics_tracker=metrics_tracker)


def load_model(model_path: str, cfg_path: str, device: str = "cpu") -> HookedTransformer:
    """
    Loads a saved model into HookedTransformer.

    Parameters
    ----------
    model_path : str
        Path to model's weights. (typically "model0.pt")
    cfg_path : str 
        Path to model's config. (typically "model_cfg.pt")
    device : str
        Device to load the model on
    """
    if not os.path.exists(model_path) and os.path.exists(cfg_path):
        raise ValueError("Path doesn't exist.")

    cfg = torch.load(cfg_path, weights_only=False, map_location=device)
    cfg.device = device

    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


if __name__ == "__main__":
    T0 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0.5]
    ])
    T1 = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0.5, 0, 0]
    ])

    dataset = MarkovData(n_gen=10000, gen_len=64, n_states=3, d_vocab=2, T_list=[T0, T1])

    if os.path.exists("./toy_transformer_checkpoints/model0.pt"):
        model = load_model("./toy_transformer_checkpoints/model0.pt", 
                         "./toy_transformer_checkpoints/model_cfg.pt")
    else:
        model = train_model(
            dataset=dataset, 
            n_epochs=5, 
            save_every=1000, 
            print_every=1000, 
            save_dir="./toy_transformer_checkpoints",
            # Enable ALL advanced metrics
            track_ngrams=True,
            ngram_orders=[2, 3],
            track_composition=True,
            track_previous_token=True,
            track_in_context=True,
            track_prefix_matching=True,
            wandb=True,
            wandb_project_name="advanced_toy_transformer"
        )

    model2 = finetune_model(
        model, dataset, 5, 
        save_dir=None,
        # Enable ALL advanced metrics for finetuning too
        track_ngrams=True,
        track_composition=True,
        track_previous_token=True,
        track_in_context=True,
        track_prefix_matching=True
    )

    # Test inference
    logits = model(torch.tensor([[0,1,1,0,1,0,0,1,1,0], 
                                [1,0,1,1,0,1,0,0,1,1], 
                                [1,0,0,1,0,0,1,0,0,1]], dtype=torch.int64))
    print(logits[:, -1])
    print(logits[:, -1].argmax(dim=-1))

    # Sample and compare
    sample, states = dataset.model.sample_sequence(max_new_tokens=40)
    preds = model(torch.tensor([sample], dtype=torch.int64)).argmax(dim=-1).flatten().tolist()

    for s, pred in zip(sample[1:], preds[:-1]):
        print(f"Actual: {s}, Predicted: {pred}")
