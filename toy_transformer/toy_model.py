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
        return {'tokens': self.data[idx]}

class MergeMarkovDatasets(Dataset):
    """
    Merges two MarkovData objects and returns a new Dataset object.

    Mixing style is the manner in which the datasets should be merged.
    `'random'` : Generations from both the datasets are randomly mixed.
    `'alternate'` : The new dataset has generations alternating from both the datasets.
    `'stack'` : Generations of the second dataset are added after generations of the first dataset.

    Note that mixing_style may play an important role in training of the model.
    """
    def __init__(
        self,
        dataset1: MarkovData,
        dataset2: MarkovData,
        mixing_style: Literal['random', 'alternate', 'stack']
        ):
        self.model1 = dataset1.model
        self.model2 = dataset2.model
        assert dataset1.d_vocab == dataset2.d_vocab, 'Vocabulary size for the datasets does not match'
        self.d_vocab = dataset1.d_vocab
        assert dataset1.gen_len == dataset2.gen_len, 'Generations lengths for the datasets do not match'
        self.gen_len = dataset1.gen_len
        
        data1 = list(zip(dataset1.data, dataset1.states))
        data2 = list(zip(dataset2.data, dataset2.states))

        if mixing_style == 'random':
            merged = data1 + data2
            np.random.shuffle(merged)
        elif mixing_style == 'alternate':
            assert len(data1) == len(data2), 'Mixing style \'alternate\' is valid only when the size of both datasets is same'
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
        return {'tokens': self.data[idx]}

def train(
    model: HookedTransformer,
    config: HookedTransformerTrainConfig,
    train_data: Dataset,
    val_data: Optional[Dataset] = None,
    eval_every: Optional[int] = None
) -> HookedTransformer:
    """
    Helper function to train an HookedTransformer model on an autoregressive language modeling task.
    (Slightly modified version of TransformerLens one.)
    Args:
        model: The model to train
        config: The training configuration
        train_data: The dataset to train on
        val_data: The dataset to use for validation
        eval_every: Number of epochs after which to run the model on `val_data`
    Returns:
        The trained model
    """
    torch.manual_seed(config.seed)
    model.train()
    if config.wandb:
        if config.wandb_project_name is None:
            config.wandb_project_name = "easy-transformer"
        wandb.init(project=config.wandb_project_name, config=vars(config))

    if config.optimizer_name in ["Adam", "AdamW"]:
        # Weight decay in Adam is implemented badly, so use AdamW instead (see PyTorch AdamW docs)
        if config.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
            )
    elif config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=(config.weight_decay if config.weight_decay is not None else 0.0),
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} not supported")

    scheduler = None
    if config.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.warmup_steps),
        )

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=len(val_data)) if val_data else None

    model = model.to(config.device)

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

            if config.wandb:
                wandb.log({"train_loss": loss.item(), "samples": samples, "epoch": epoch})

        if config.print_every is not None and epoch % config.print_every == 0:
            print(f"Epoch {epoch} Samples {samples} Step {step} Training Loss {loss.item()}")

        if (
            config.save_every is not None
            and epoch % config.save_every == 0
            and config.save_dir is not None
        ):
            torch.save(model.state_dict(), f"{config.save_dir}/model_{epoch}.pt")

        if val_dataloader and eval_every is not None and epoch % eval_every == 0:
            for data in val_dataloader:
                model.eval()
                tokens = data["tokens"].to(config.device)
                with torch.no_grad():
                    loss = model(tokens, return_type="loss")
                if config.wandb:
                    wandb.log({"val_loss": loss.item(), "epoch": epoch})
                print(f"Epoch {epoch} Validation Loss {loss}")

    return model

def train_model(
    # Dataset
    dataset: MarkovData | MergeMarkovDatasets,

    # Transformer Architecture
    n_layers: int = 4,
    d_model: int = 64,
    n_heads: int = 1,
    d_head: int = 8,
    attn_only: bool = False,
    d_mlp: int = 256,
    act_fn: Literal['relu', 'gelu', 'silu', 'gelu_new', 'solu_ln', 'gelu_fast'] = 'relu',
    normalization_type: Literal['LN', 'LNPre', 'RMS', 'RMSPre'] = None,
    positional_embedding_type: Literal['standard', 'rotary', 'shortformer'] = 'standard',

    # Training Hyperparameters
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-2,
    optimizer_name: Literal['Adam', 'AdamW', 'SGD'] = 'SGD',
    wandb: bool = False,
    wandb_project_name: Optional[str] = None,

    # System / IO
    device: str = "cpu",
    seed: int = 42,
    save_every: int = 1,
    save_dir: str = "./checkpoints",
    print_every: int = 1,
    eval_every: int = 1,
    val_frac: float = 0.2
) -> HookedTransformer:
    """
    Train a HookedTransformer on sequences generated from a Mealy Markov model.

    This function constructs a HookedTransformer model with the given 
    architecture and optimization hyperparameters, and trains it on 
    sequences generated from a custom Markov process dataset.

    Parameters
    ----------
    dataset : MarkovData or MergeMarkovDatasets
        Training dataset containing token sequences generated from a 
        Mealy-Markov process.

    n_layers : int
        Number of transformer layers.
    d_model : int
        Dimension of the model (embedding and hidden sizes).
    n_heads : int
        Number of attention heads.
    d_head : int
        Dimension per attention head.
    attn_only : bool
        Whether the transformer is attention-only, without any MLP blocks.
    d_mlp : int
        Dimension of the feedforward hidden layer.
    act_fn : {'relu', 'gelu', 'silu', 'gelu_new', 'solu_ln', 'gelu_fast'}
        Activation function used in MLP layers.
    normalization_type : {'LN', 'LNPre', 'RMS', 'RMSPre'}
        Normalization strategy applied in transformer layers. Defaults to no normalization
    positional_embedding_type : {'standard', 'rotary', 'shortformer'}
        Type of positional embeddings used in the model.

    n_epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate for optimization.
    optimizer_name : {'Adam', 'AdamW', 'SGD'}
        Optimizer to use.
    wandb : bool
        Whether to use wandb to log training
    wandb_project_name : str, optional
        Name for wandb project, defaults to 'easy-transformer'
    
    device : str
        Device where the model will be trained (e.g., ``"cpu"``, ``"cuda"``).
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
        Fraction of `dataset` to be used as validation dataset

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

    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if save_dir is not None:
        torch.save(cfg, save_dir + '/model_cfg.pt')

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

    # Train-val split
    if val_frac:
        train_size = int(len(dataset) * (1 - val_frac))
        indices = torch.randperm(len(dataset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_data = Subset(dataset, train_indices)
        val_data = Subset(dataset, val_indices)
        return train(model, train_cfg, train_data, val_data, eval_every)
    else:
        return train(model, train_cfg, dataset)

def finetune_model(
    model: HookedTransformer,
    dataset: MarkovData | MergeMarkovDatasets,
    n_epochs: int,
    batch_size: int = 64,
    lr: float = 1e-2,
    optimizer_name: Literal['Adam', 'AdamW', 'SGD'] = 'SGD',
    wandb: bool = False,
    wandb_project_name: Optional[str] = None,
    device: str = "cpu",
    seed: int = 42,
    save_every: int = 1,
    save_dir: str = "./checkpoints",
    print_every: int = 1,
    eval_every: int = 1,
    val_frac: float = 0.2
) -> HookedTransformer:
    """
    Finetune a pretrained HookedTransformer on sequences generated from a Mealy Markov model.

    Parameters
    ----------
    model : HookedTransformer
        A pre-trained model to finetune.
    dataset : MarkovData or MergeMarkovDatasets
        Training dataset containing token sequences generated from a 
        Mealy-Markov process.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate for optimization.
    optimizer_name : {'Adam', 'AdamW', 'SGD'}
        Optimizer to use.
    wandb : bool
        Whether to use wandb to log training
    wandb_project_name : str, optional
        Name for wandb project, defaults to 'easy-transformer'
    device : str
        Device where the model will be trained (e.g., ``"cpu"``, ``"cuda"``).
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
        Fraction of `dataset` to be used as validation dataset

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
        return train(model, cfg, train_data, val_data, eval_every)
    else:
        return train(model, cfg, dataset)

def load_model(model_path: str, cfg_path: str, device: str = 'cpu') -> HookedTransformer:
    '''
    Loads a saved model into HookedTransformer.
    
    Parameters
    ----------
    model_path : str
        Path to model's weights. (typically `model_0.pt`)
    cfg_path : str
        Path to model's config. (typically `model_cfg.pt`)
    device : str
        Device to load the model on
    '''
    if not (os.path.exists(model_path) and os.path.exists(cfg_path)):
        raise ValueError('Path doesn\'t exist.')
    cfg = torch.load(cfg_path, weights_only=False, map_location=device)
    cfg.device = device
    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

if __name__ == '__main__':
    T0 = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0.5, 0, 0]
    ])
    T1 = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0.5, 0, 0]
    ])
    dataset = MarkovData(n_gen=10000, gen_len=64, n_states=3, d_vocab=2, T_list=[T0, T1])

    if os.path.exists('toy_transformer/checkpoints/model_0.pt'):
        model = load_model('toy_transformer/checkpoints/model_0.pt',
                           'toy_transformer/checkpoints/model_cfg.pt')
    else:
        model = train_model(
            dataset=dataset,
            n_epochs=5,
            save_every=1000,
            print_every=1000,
            save_dir='toy_transformer/checkpoints'
        )

    model_2 = finetune_model(model, dataset, 5, save_dir=None)

    logits = model(torch.tensor([[0,1,1,0,1,0,0,1,1,0],
                                 [1,0,1,1,0,1,0,0,1,1],
                                 [1,0,0,1,0,0,1,0,0,1]], dtype=torch.int64))
    print(logits[:, -1, :])
    print(logits[:, -1, :].argmax(dim=-1))
    # Ground truth values: [1, 0, R]

    print()
    sample, states = dataset.model.sample_sequence(max_new_tokens=40)
    preds = model(torch.tensor(sample, dtype=torch.int64)).argmax(dim=-1).flatten().tolist()
    for s, pred in zip(sample[1:], preds[:-1]):
        print(f'Actual: {s}, Predicted: {pred}')
