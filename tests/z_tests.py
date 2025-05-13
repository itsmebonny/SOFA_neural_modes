import argparse
import os
import sys
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

# Add the parent directory of 'training' to the Python path
# to allow importing from train_sofa
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir) # This should be SOFA_neural_modes
training_module_path = os.path.join(parent_dir, 'training')
if training_module_path not in sys.path:
    sys.path.insert(0, training_module_path)

try:
    from training.train_sofa import Routine, load_config
except ImportError as e:
    print(f"Error importing from train_sofa: {e}")
    print(f"Please ensure train_sofa.py is accessible and all its dependencies are installed.")
    print(f"Current sys.path includes: {sys.path}")
    sys.exit(1)

torch.set_default_dtype(torch.float64)

def generate_random_z_samples(num_samples, latent_dim, device, current_scale_for_sampling):
    """
    Generates random latent vectors 'z' using the logistic decay scaling,
    similar to the training process.
    """
    # Hyperparameters for logistic decay (should match training)
    min_amplitude_factor = 0.05
    inflection_point_mode_ratio = 0.1
    decay_steepness = 1.0
    L = latent_dim

    max_amplitude_mode0 = current_scale_for_sampling
    min_amplitude_higher_modes = max_amplitude_mode0 * min_amplitude_factor
    
    mode_indices = torch.arange(L, device=device, dtype=torch.float64)
    inflection_point_abs = L * inflection_point_mode_ratio
    
    exponent_term = decay_steepness * (mode_indices - inflection_point_abs)
    logistic_decay_value = 1.0 / (1.0 + torch.exp(exponent_term))
    
    individual_mode_scales = min_amplitude_higher_modes + \
                             (max_amplitude_mode0 - min_amplitude_higher_modes) * logistic_decay_value
    
    z_unit_range = torch.rand(num_samples, L, device=device) * 2.0 - 1.0
    z_samples = z_unit_range * individual_mode_scales.unsqueeze(0)
    
    return z_samples

def main():
    parser = argparse.ArgumentParser(description='Evaluate Model on Random Z vectors')
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a specific model checkpoint. Defaults to best_sofa.pt in config.')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of random z samples to evaluate')
    parser.add_argument('--sampling_scale', type=float, default=100.0, # Matches deformation_scale_final
                        help='The "current_scale" to use for z sampling range (max amplitude for mode 0)')
    args = parser.parse_args()

    # Construct absolute path for config if it's relative
    if not os.path.isabs(args.config):
        config_path = os.path.join(parent_dir, args.config)
    else:
        config_path = args.config
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)

    print("Initializing Routine...")
    engine = Routine(cfg)
    engine.model.eval() # Set model to evaluation mode
    print("Routine initialized.")

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.isabs(checkpoint_path): # Handle relative path for checkpoint
             checkpoint_path = os.path.join(parent_dir, checkpoint_path)
    else:
        checkpoint_dir = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')
        # Construct absolute path for checkpoint_dir if it's relative to project root
        if not os.path.isabs(checkpoint_dir):
            checkpoint_dir = os.path.join(parent_dir, checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, 'best_sofa.pt')

    if os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        engine.load_checkpoint(checkpoint_path)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using an untrained or randomly initialized model.")

    L = engine.latent_dim
    linear_modes_eval = engine.linear_modes[:, :L].to(engine.device) # Ensure it's on the correct device

    print(f"Generating {args.num_samples} random z samples with max mode 0 amplitude: {args.sampling_scale}...")
    z_batch = generate_random_z_samples(args.num_samples, L, engine.device, args.sampling_scale)

    delta_energies = []
    z_norms = []

    print("Evaluating samples...")
    E_linear_values = []
    E_nonlinear_values = []
    with torch.no_grad():
        for i in range(args.num_samples):
            z = z_batch[i:i+1, :] # Keep batch dimension for model and energy_calculator

            # 1. Linear displacement
            l = torch.matmul(z, linear_modes_eval.T)

            # 2. Neural correction
            y = engine.model(z) # Model should handle batch_size=1

            # 3. Total displacement
            u_total = l + y

            # 4. Compute energies
            # Ensure inputs to energy_calculator are correctly shaped (batch_size, num_dofs)
            E_linear_batch = engine.energy_calculator(l)
            E_nonlinear_batch = engine.energy_calculator(u_total)
            
            E_linear = E_linear_batch[0] # Get scalar energy
            E_nonlinear = E_nonlinear_batch[0] # Get scalar energy

            delta_E = E_nonlinear - E_linear

            delta_E_rel = (E_nonlinear - E_linear)/E_linear if E_linear.abs() > 1e-9 else torch.tensor(0.0, device=E_linear.device) # Avoid division by zero
            norm_z = torch.norm(z.squeeze()) # Squeeze to get 1D vector for norm

            delta_energies.append(delta_E.item())
            z_norms.append(norm_z.item())
            E_linear_values.append(E_linear.item())
            E_nonlinear_values.append(E_nonlinear.item())

            if (i + 1) % (args.num_samples // 10 if args.num_samples >= 10 else 1) == 0:
                print(f"  Processed {i+1}/{args.num_samples} samples...")

    print("Evaluation complete.")

    # Create save directory if it doesn't exist
    save_dir_name = "z_debug"
    save_dir_path = os.path.join(script_dir, save_dir_name)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
        print(f"Created directory: {save_dir_path}")

    # Plot 1: Absolute Energy Difference (delta_E)
    plt.figure(figsize=(10, 6))
    plt.scatter(z_norms, delta_energies, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.xlabel("Norm of z (||z||_2)")
    plt.ylabel("Absolute Energy Difference (ΔE = E_nonlinear - E_linear)")
    plt.title(f"Absolute Energy Difference vs. Norm of Latent Vector z (N={args.num_samples})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_filename_absolute = "energy_diff_absolute_vs_z_norm.png"
    save_path_absolute = os.path.join(save_dir_path, plot_filename_absolute)
    try:
        plt.savefig(save_path_absolute)
        print(f"Absolute energy difference plot saved to {save_path_absolute}")
    except Exception as e:
        print(f"Error saving absolute energy difference plot: {e}")
    plt.close()

    # Plot 2: Relative Energy Difference (delta_E_rel)
    delta_energies_rel = [
        (E_nonlinear_values[i] - E_linear_values[i]) / E_linear_values[i]
        if abs(E_linear_values[i]) > 1e-9 else 0.0
        for i in range(len(E_linear_values))
    ]
    plt.figure(figsize=(10, 6))
    plt.scatter(z_norms, delta_energies_rel, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.xlabel("Norm of z (||z||_2)")
    plt.ylabel("Relative Energy Difference (ΔE/E_linear)")
    plt.title(f"Relative Energy Difference vs. Norm of Latent Vector z (N={args.num_samples})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_filename_relative = "energy_diff_relative_vs_z_norm.png"
    save_path_relative = os.path.join(save_dir_path, plot_filename_relative)
    try:
        plt.savefig(save_path_relative)
        print(f"Relative energy difference plot saved to {save_path_relative}")
    except Exception as e:
        print(f"Error saving relative energy difference plot: {e}")
    plt.close()

    # Plot 3: Linear and Nonlinear Energies vs z_norm
    plt.figure(figsize=(10, 6))
    plt.scatter(z_norms, E_linear_values, alpha=0.6, edgecolors='b', linewidth=0.5, label='E_linear')
    plt.scatter(z_norms, E_nonlinear_values, alpha=0.6, edgecolors='r', linewidth=0.5, label='E_nonlinear')
    plt.xlabel("Norm of z (||z||_2)")
    plt.ylabel("Energy")
    plt.title(f"Linear and Nonlinear Energies vs. Norm of Latent Vector z (N={args.num_samples})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_filename_energies = "energy_linear_nonlinear_vs_z_norm.png"
    save_path_energies = os.path.join(save_dir_path, plot_filename_energies)
    try:
        plt.savefig(save_path_energies)
        print(f"Linear and nonlinear energy plot saved to {save_path_energies}")
    except Exception as e:
        print(f"Error saving linear/nonlinear energy plot: {e}")
    plt.close()

if __name__ == '__main__':
    main()