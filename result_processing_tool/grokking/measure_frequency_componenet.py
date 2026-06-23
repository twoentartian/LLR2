import sys, os

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import util
import py_src.ml_setup as ml_setup

# Token layout (from ArithmeticTokenizer.get_tokens):
#   [0]       <|eos|>
#   [1]       =
#   [2..21]   sorted operator tokens (20 operators)
#   [22..118] arithmetic numbers 0..96  (P=97 numbers)
#   [119+]    s5 permutations

def compute_fourier_components(state_dict, token_offset, P=97):
    """
    Compute Fourier components of W_E and W_L as in Figure 3 of
    Nanda et al. (2023), accounting for the correct token index offset.

    Token layout:  [<EOS>(0), =(1), *operators*(2-21), 0..96(22-118), s5perms(119+)]
    Input format:  [<EOS>, a, +, b, =, answer, <EOS>]
                    pos 0   1  2  3  4    5       6
    """

    def to_numpy(key):
        return state_dict[key].detach().cpu().float().numpy()

    # ------------------------------------------------------------------ #
    # 1. Extract weights
    # ------------------------------------------------------------------ #
    W_E      = to_numpy('embedding.weight')                   # (2000, 128)
    W_U      = to_numpy('linear.weight')                      # (2000, 128)
    W_out_b0 = to_numpy('decoder.blocks.0.ffn.ffn.2.weight')  # (128, 512)
    W_out_b1 = to_numpy('decoder.blocks.1.ffn.ffn.2.weight')  # (128, 512)
    pos_enc  = to_numpy('position_encoding')                  # (50, 128)

    d_model = min(W_E.shape)

    # Normalise orientations: W_E/W_U -> (vocab, d_model), W_out -> (d_model, n)
    if W_E.shape[1] != d_model:
        W_E = W_E.T
    if W_U.shape[1] != d_model:
        W_U = W_U.T
    if W_out_b0.shape[0] != d_model:
        W_out_b0 = W_out_b0.T
    if W_out_b1.shape[0] != d_model:
        W_out_b1 = W_out_b1.T

    # ------------------------------------------------------------------ #
    # 2. Slice to arithmetic tokens only using the correct offset
    #    Numbers 0..96 live at vocab indices 22..118
    # ------------------------------------------------------------------ #
    arith_slice = slice(token_offset, token_offset + P)  # [22:119]

    W_E_arith = W_E[arith_slice, :]   # (97, 128)
    W_U_arith = W_U[arith_slice, :]   # (97, 128)

    # ------------------------------------------------------------------ #
    # 3. Position-adjusted embedding
    #    'a' is always at position 1 in [<EOS>, a, +, b, =, answer, <EOS>]
    # ------------------------------------------------------------------ #
    W_E_pos = W_E_arith + pos_enc[1]   # (97, 128)

    # ------------------------------------------------------------------ #
    # 4. Neuron-logit map: W_L = W_U_arith @ W_out  ->  (97, 1024)
    # ------------------------------------------------------------------ #
    W_out = np.concatenate([W_out_b0, W_out_b1], axis=1)  # (128, 1024)
    W_L   = W_U_arith @ W_out                              # (97, 1024)

    # ------------------------------------------------------------------ #
    # 5. Build DFT matrices of shape (P, P)
    # ------------------------------------------------------------------ #
    freqs     = np.arange(P, dtype=np.float32)
    positions = np.arange(P, dtype=np.float32)
    angles    = 2 * np.pi * np.outer(freqs, positions) / P

    F_cos = np.cos(angles) / P
    F_sin = np.sin(angles) / P

    # ------------------------------------------------------------------ #
    # 6. Compute norms of Fourier components
    # ------------------------------------------------------------------ #
    W_E_cos_norms     = np.linalg.norm(F_cos @ W_E_arith, axis=1)
    W_E_sin_norms     = np.linalg.norm(F_sin @ W_E_arith, axis=1)
    W_E_pos_cos_norms = np.linalg.norm(F_cos @ W_E_pos,   axis=1)
    W_E_pos_sin_norms = np.linalg.norm(F_sin @ W_E_pos,   axis=1)
    W_L_cos_norms     = np.linalg.norm(F_cos @ W_L,       axis=1)
    W_L_sin_norms     = np.linalg.norm(F_sin @ W_L,       axis=1)

    # ------------------------------------------------------------------ #
    # 7. Restrict to k = 0..P//2 (symmetric spectrum)
    # ------------------------------------------------------------------ #
    half = P // 2 + 1
    s    = slice(0, half)

    return {
        'freqs':             np.arange(half),
        'W_E_cos_norms':     W_E_cos_norms[s],
        'W_E_sin_norms':     W_E_sin_norms[s],
        'W_E_pos_cos_norms': W_E_pos_cos_norms[s],
        'W_E_pos_sin_norms': W_E_pos_sin_norms[s],
        'W_L_cos_norms':     W_L_cos_norms[s],
        'W_L_sin_norms':     W_L_sin_norms[s],
    }


def plot_fourier_components(results, offset, modulus, save_path):
    freqs = results['freqs']
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    ax = axes[0]
    ax.plot(freqs, results['W_E_sin_norms'], label='sin', color='steelblue')
    ax.plot(freqs, results['W_E_cos_norms'], label='cos', color='darkorange')
    ax.set_title(f'Fourier Components of W_E (arith tokens [{offset}:{offset+modulus}])')
    ax.set_xlabel('Frequency k')
    ax.set_ylabel('Norm of Fourier Component')
    ax.legend()

    ax = axes[1]
    ax.plot(freqs, results['W_E_pos_sin_norms'], label='sin', color='steelblue')
    ax.plot(freqs, results['W_E_pos_cos_norms'], label='cos', color='darkorange')
    ax.set_title("Fourier Components of W_E + pos_enc[1] (position of 'a')")
    ax.set_xlabel('Frequency k')
    ax.set_ylabel('Norm of Fourier Component')
    ax.legend()

    ax = axes[2]
    ax.plot(freqs, results['W_L_sin_norms'], label='sin', color='steelblue')
    ax.plot(freqs, results['W_L_cos_norms'], label='cos', color='darkorange')
    ax.set_title('Fourier Components of Neuron-Logit Map')
    ax.set_xlabel('Frequency k')
    ax.set_ylabel('Norm of Fourier Component')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ------------------------------------------------------------------ #
# Example usage
# ------------------------------------------------------------------ #
# state_dict = torch.load('checkpoint.pt', map_location='cpu')
# results = compute_fourier_components(model, state_dict, P=113)
# plot_fourier_components(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("calculate the fourier frequency component of model weights")
    parser.add_argument("path", type=Path, help="File or directory path containing .model.pt files.")
    parser.add_argument("--offset", type=int, default=2, help="Index of number '0' in the vocabulary")
    parser.add_argument("-m", "--modulus", type=int, default=97, help="the modulus, default=97")
    args = parser.parse_args()

    modulus = args.modulus
    p = args.path.expanduser().resolve()
    paths = (sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir()
             else [p] if p.is_file()
    else (_ for _ in ()).throw(FileNotFoundError(p)))

    global_model_name, global_dataset_name, global_ml_setup = None, None, None
    for p in paths:
        print(f"processing: {p}")
        model_state, model_name, dataset_name = util.load_model_state_file(p)
        if global_model_name is None:
            global_model_name = model_name
        else:
            assert global_model_name == model_name
        if global_dataset_name is None:
            global_dataset_name = dataset_name
        else:
            assert global_dataset_name == dataset_name
        if global_ml_setup is None:
            global_ml_setup = ml_setup.get_ml_setup_from_config(global_model_name, global_dataset_name, pytorch_preset_version=0, device=torch.device('cpu'))

        offset = args.offset
        result = compute_fourier_components(model_state, offset, P=modulus)
        save_folder = f"{str(Path(p).parent)}_fourier_components"
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{str(Path(p).name)}.pdf")
        plot_fourier_components(result, offset, modulus, save_path)
