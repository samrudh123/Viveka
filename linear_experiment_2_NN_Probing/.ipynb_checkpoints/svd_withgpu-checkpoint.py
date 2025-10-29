import argparse
import glob
import os
import torch as t
from tqdm import tqdm

def perform_global_svd(activations_dir, svd_dim, layer_indices, device):
    """
    Performs SVD for each specified layer on its full activation data,
    saving the projection matrix (V_T) and also saving merged activations
    projected into the top `svd_dim` singular vectors.
    """
    print(f"--- Starting Global SVD stage for layers: {layer_indices} ---")
    print(f'SVD dim :{svd_dim}')
    #if svd_dim ==2304:#to be changed for other models
     #   print(f"Preparing full dim thin SVD for layers {layer_indices}")
        
    # Directory for projection matrices
    svd_dir = os.path.join(os.path.dirname(activations_dir), f'svd_components_{svd_dim}')
    os.makedirs(svd_dir, exist_ok=True)
    svd_eigen_dir = os.path.join(os.path.dirname(activations_dir), f'eigenvalues_{svd_dim}')
    os.makedirs(svd_eigen_dir, exist_ok=True)
    

    # Directory for projected activations
    projected_dir = os.path.join(os.path.dirname(activations_dir), f'activations_svd_{svd_dim}',
                                 os.path.basename(activations_dir))
    os.makedirs(projected_dir, exist_ok=True)

    for l_idx in tqdm(layer_indices, desc="Performing SVD per layer"):
        projection_matrix_path = os.path.join(svd_dir, f"projection_matrix_layer_{l_idx}.pt")
        eigen_dir_path = os.path.join(svd_eigen_dir, f"eigen_values_layer_{l_idx}.pt")
        # ---- Load merged activations ----
        merged_path = os.path.join(activations_dir, f"layer_{l_idx}_balanced.pt" )
        if not os.path.exists(merged_path):
            print(f"⚠️ No merged activation file found for layer {l_idx} at '{merged_path}'. Skipping.")
            continue

        data = t.load(merged_path, map_location="cpu")
        full_layer_activations = data["activations"].to(device)
        labels = data["labels"]
        orig_dtype = full_layer_activations.dtype

        # ---- Ensure float32 for SVD ----
        if orig_dtype in (t.float16, t.bfloat16):
            print(f"Converting layer {l_idx} activations from {orig_dtype} to float32 for SVD.")
            full_layer_activations = full_layer_activations.to(t.float32)

        # ---- Run SVD ----
        print(f"Running SVD on {full_layer_activations.shape[0]} activations for layer {l_idx}...")
        try:
            _, S, Vh = t.linalg.svd(full_layer_activations, full_matrices=False)
            t.cuda.empty_cache()
        except t.cuda.OutOfMemoryError:
            print(f"CUDA OOM on layer {l_idx}. Falling back to CPU.")
            _, S, Vh = t.linalg.svd(full_layer_activations.cpu(), full_matrices=False)
            Vh = Vh.to(device)
            S = S.to(device)

        projection_matrix = Vh[:svd_dim, :]
        eigvals = S**2
        t.save(projection_matrix.cpu(), projection_matrix_path)
        t.save(eigvals.cpu(), eigen_dir_path)
        print(f"Saved SVD projection matrix and eigen values for layer {l_idx} to {projection_matrix_path}")

        # ---- Project *all* activations in one go ----
        if projection_matrix.dtype != t.float32:
            projection_matrix = projection_matrix.to(t.float32)

        projected_activations = (projection_matrix @ full_layer_activations.T).T
        projected_activations = projected_activations.to(orig_dtype)

        # ---- Save merged projected file ----
        save_name = f"layer_{l_idx}_balanced_svd_processed.pt"
        save_path = os.path.join(projected_dir, save_name)

        t.save({'activations': projected_activations.cpu(), 'labels': labels}, save_path)
        print(f"Saved projected activations for layer {l_idx} to {save_path}")
        t.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform SVD on saved activations to find principal components.")
    parser.add_argument('--probe_output_dir', type=str, default='probes_data', help="Root directory where activation data is stored.")
    parser.add_argument('--model_repo_id', type=str, required=True, help="Hugging Face model ID, used to find the correct activation folder.")
    parser.add_argument('--svd_layers', nargs='+', type=int, required=True, help="List of layers to perform SVD on. Use -1 for all.")
    parser.add_argument('--svd_dim', type=int, default=576, help="Dimension to reduce activations to via SVD.")
    parser.add_argument('--device', type=str, default='cuda' if t.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    activations_dir = os.path.join(args.probe_output_dir, 'activations', args.model_repo_id.replace('/', '_'))    
    if -1 == args.svd_layers:
        layer_indices = [i for i in range(26)]
    else:
        layer_indices = args.svd_layers

    perform_global_svd(activations_dir, args.svd_dim, layer_indices, args.device)
