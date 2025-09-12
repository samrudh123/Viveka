import argparse
import glob
import os
import torch as t
from tqdm import tqdm

def perform_global_svd(activations_dir, svd_dim, layer_indices, device):
    """
    Performs SVD for each specified layer on its full activation data,
    saving the projection matrix (V_T) and also saving per-statement
    activations projected into the top `svd_dim` singular vectors.
    """
    print(f"--- Starting Global SVD stage for layers: {layer_indices} ---")
    # Directory for projection matrices
    svd_dir = os.path.join(os.path.dirname(activations_dir), 'svd_components')
    os.makedirs(svd_dir, exist_ok=True)
    # Directory for projected activations
    projected_dir = os.path.join(os.path.dirname(activations_dir), 'activations_svd',
                                 os.path.basename(activations_dir))
    os.makedirs(projected_dir, exist_ok=True)

    for l_idx in tqdm(layer_indices, desc="Performing SVD per layer"):
        projection_matrix_path = os.path.join(svd_dir, f"projection_matrix_layer_{l_idx}.pt")
        if os.path.exists(projection_matrix_path):
            print(f"SVD projection matrix for layer {l_idx} already exists. Loading it.")
            projection_matrix = t.load(projection_matrix_path).to(device)
        else:
            # Load all activation files for the current layer
            file_pattern = os.path.join(activations_dir, f'layer_{l_idx}_stmt_*.pt')
            all_files = glob.glob(file_pattern)
            if not all_files:
                print(f"Warning: No activation files found for layer {l_idx} at '{file_pattern}'. Skipping.")
                continue
                
            # Concatenate all activations
            activations_list = [t.load(fname)['activations'] for fname in tqdm(all_files, desc=f"Loading L{l_idx} files", leave=False)]
            full_layer_activations = t.cat(activations_list, dim=0).to(device)

            # Save original dtype
            orig_dtype = full_layer_activations.dtype

            # Convert to float32 for SVD if needed
            if orig_dtype in  (t.float16,t.bfloat16):
                print(f"Converting layer {l_idx} activations from bfloat16 to float32 for SVD.")
                full_layer_activations = full_layer_activations.to(t.float32)

            # Run SVD
            print(f"Running SVD on {full_layer_activations.shape[0]} activations for layer {l_idx}...")
            try:
                _, _, Vh = t.linalg.svd(full_layer_activations, full_matrices=False)
            except t.cuda.OutOfMemoryError:
                print(f"CUDA OOM on layer {l_idx}. Falling back to CPU.")
                _, _, Vh = t.linalg.svd(full_layer_activations.cpu(), full_matrices=False)
                Vh = Vh.to(device)

            projection_matrix = Vh[:svd_dim, :]
            t.save(projection_matrix.cpu(), projection_matrix_path)
            print(f"Saved SVD projection matrix for layer {l_idx} to {projection_matrix_path}")

            # Project individual activations
            for fname in tqdm(all_files, desc=f"Projecting L{l_idx} activations", leave=False):
                data = t.load(fname)
                raw_activations = data['activations'].to(device)

                # Ensure projection is done in float32
                if raw_activations.dtype != t.float32:
                    raw_activations = raw_activations.to(t.float32)

                if projection_matrix.dtype != t.float32:
                    projection_matrix = projection_matrix.to(t.float32)    

                projected_activations = (projection_matrix @ raw_activations.T).T

                # Cast back to original dtype before saving
                projected_activations = projected_activations.to(orig_dtype)

                stmt_num = fname.split('_stmt_')[1].split('.pt')[0]
                save_name = f"layer_{l_idx}_stmt_{stmt_num}_svd_processed.pt"
                save_path = os.path.join(projected_dir, save_name)

                t.save({'activations': projected_activations.cpu(), 'labels': data['labels']}, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform SVD on saved activations to find principal components.")
    parser.add_argument('--probe_output_dir', type=str, default='probes_data', help="Root directory where activation data is stored.")
    parser.add_argument('--model_repo_id', type=str, required=True, help="Hugging Face model ID, used to find the correct activation folder.")
    parser.add_argument('--svd_layers', nargs='+', type=int, required=True, help="List of layers to perform SVD on. Use -1 for all.")
    parser.add_argument('--svd_dim', type=int, default=576, help="Dimension to reduce activations to via SVD.")
    parser.add_argument('--device', type=str, default='cuda' if t.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    activations_dir = os.path.join(args.probe_output_dir, 'activations', args.model_repo_id.replace('/', '_'))    
    if -1 in args.svd_layers:
        args.svd_layers = [i for i in range(26)]
    else:
        layer_indices = args.svd_layers

    perform_global_svd(activations_dir, args.svd_dim, layer_indices, args.device)
