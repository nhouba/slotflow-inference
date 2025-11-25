import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import corner
import seaborn as sns


def count_parameters(model):
    """Print and return the number of trainable parameters in the model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total


@torch.no_grad()
def plot_dataset_consistency(
    dataset, idx: int, show_fft: bool = True, overlay_components: bool = False
):
    """
    Visualize and sanity-check that coarse_input / fine_input match the reference_full signal.

    Args
    ----
    dataset : MultiSinusoidDataset with mode="debug" or "inference"
    idx     : sample index
    show_fft: also compare spectra (reference vs coarse; reference-cropped vs fine)
    overlay_components: overlay per-component coarse-view sinusoids in time domain
    """
    sample = dataset[idx]

    if len(sample) == 7:
        (
            coarse_input,
            fine_input,
            K,
            comps_coarse,
            params,
            noise_scale,
            reference_full,
        ) = sample
    else:
        raise ValueError(
            "Dataset must be created with mode='debug' or 'inference' to include reference_full."
        )

    # move to cpu, flatten
    def _cpu1d(x):
        return x.detach().float().view(-1).cpu()

    coarse_input = _cpu1d(coarse_input)
    fine_input = _cpu1d(fine_input)
    reference_full = _cpu1d(reference_full)
    comps_coarse = comps_coarse.detach().float().cpu()

    # --- time axes ---
    t_master = dataset.t_master.detach().float().cpu()
    t_coarse = torch.linspace(0.0, dataset.tEnd_long, dataset.num_samples_long)
    t_fine = torch.linspace(0.0, dataset.tEnd_short, len(fine_input))

    # --- resample reference_full for comparison ---
    reference_to_coarse = F.interpolate(
        reference_full.view(1, 1, -1),
        size=len(coarse_input),
        mode="linear",
        align_corners=False,
    ).view(-1)

    dt_master = float(t_master[1] - t_master[0])
    n_fine_master = int(round(dataset.tEnd_short / dt_master))
    n_fine_master = min(n_fine_master, len(reference_full))
    reference_cropped = reference_full[:n_fine_master]
    reference_to_fine = F.interpolate(
        reference_cropped.view(1, 1, -1),
        size=len(fine_input),
        mode="linear",
        align_corners=False,
    ).view(-1)

    # --- numeric errors ---
    err_coarse = torch.linalg.vector_norm(
        coarse_input - reference_to_coarse
    ) / torch.linalg.vector_norm(reference_to_coarse).clamp_min(1e-9)
    err_fine = torch.linalg.vector_norm(
        fine_input - reference_to_fine
    ) / torch.linalg.vector_norm(reference_to_fine).clamp_min(1e-9)
    corr_coarse = torch.corrcoef(torch.stack([coarse_input, reference_to_coarse]))[
        0, 1
    ].item()
    corr_fine = torch.corrcoef(torch.stack([fine_input, reference_to_fine]))[
        0, 1
    ].item()

    print(f"[sample {idx}] K={K}, noise_std={noise_scale:.3f}")
    print(f"  coarse rel.L2={err_coarse:.3e}, corr={corr_coarse:.6f}")
    print(f"  fine   rel.L2={err_fine:.3e}, corr={corr_fine:.6f}")

    # --- plots ---
    # 1) Reference vs Coarse
    plt.figure(figsize=(11, 4))
    plt.plot(
        t_master,  # [t_master <= 10],
        reference_full,  # [: len(t_master[t_master <= 10])],
        label="reference_full (high-res)",
        alpha=0.6,
    )
    plt.plot(
        t_coarse,  # [t_coarse <= 10],
        coarse_input,  # [: len(t_coarse[t_coarse <= 10])],
        label="coarse_input (long/low-res)",
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time [s]")
    plt.ylabel("Signal")
    plt.title(f"Reference vs Coarse (sample {idx})")
    plt.tight_layout()

    # 2) Reference-cropped vs Fine
    plt.figure(figsize=(11, 4))
    plt.plot(
        torch.linspace(0.0, dataset.tEnd_short, len(reference_to_fine)),
        reference_to_fine,
        label="reference_full → fine window",
        alpha=0.6,
    )
    plt.plot(t_fine, fine_input, label="fine_input (short/high-res)")
    if overlay_components:
        for k in range(min(K, comps_coarse.shape[0])):
            plt.plot(t_coarse, comps_coarse[k], "--", alpha=0.4)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time [s]")
    plt.ylabel("Signal")
    plt.title(f"Reference vs Fine (sample {idx})")
    plt.tight_layout()

    # 3) Optional spectra
    if show_fft:

        def rfft_mag(x, fs):
            X = torch.fft.rfft(x)
            f = torch.fft.rfftfreq(len(x), d=1.0 / fs)
            return f, (X.abs() ** 2) / len(x)

        dt_coarse = dataset.tEnd_long / (len(coarse_input) - 1)
        fM, PM = rfft_mag(reference_full, 1.0 / dt_master)
        fC, PC = rfft_mag(coarse_input, 1.0 / dt_coarse)

        plt.figure(figsize=(11, 4))
        plt.semilogy(fM, PM + 1e-16, label="reference_full")
        plt.semilogy(fC, PC + 1e-16, label="coarse_input")
        plt.xlim(0, min(fM[-1], fC[-1]))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power")
        plt.title(f"Spectrum: Reference vs Coarse (sample {idx})")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        fs_fine = len(fine_input) / dataset.tEnd_short
        fRf, PRf = rfft_mag(reference_to_fine, fs_fine)
        fF, PF = rfft_mag(fine_input, fs_fine)

        plt.figure(figsize=(11, 4))
        plt.semilogy(fRf, PRf + 1e-16, label="reference_full → fine window")
        plt.semilogy(fF, PF + 1e-16, label="fine_input")
        plt.xlim(0, fF[-1])
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power")
        plt.title(f"Spectrum: Reference vs Fine (sample {idx})")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

    plt.show()


def plot_latent_space(
    model,
    data_loader,
    device="cpu",
    method="tsne",
    num_samples=100,
    plot_3d=False,
    latent_key="h_embed",
):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import random
    import torch
    import numpy as np

    model.eval()
    model.to(device)
    latent_vectors = []
    true_ks_list = []

    dataset = data_loader.dataset
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # ---------------------------
    # Collect latent representations
    # ---------------------------
    for idx in indices:
        item = dataset[idx]

        if len(item) == 6:
            x_long, x_short, true_k, _, _, noise_std_true = item
            mode_str = "(data generation in inference mode)"
        elif len(item) == 7:
            x_long, x_short, true_k, _, _, noise_std_true, _ = item
            mode_str = "(data generation in train mode)"
        else:
            raise ValueError(f"Unexpected number of outputs: {len(item)}")

        x_long = x_long.unsqueeze(0).to(device)
        x_short = x_short.unsqueeze(0).to(device)

        output = model(x_long, x_short)
        z = output[latent_key].detach().cpu()  # latent representation

        latent_vectors.append(z)
        true_ks_list.append(int(true_k))

    latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
    true_ks_list = np.array(true_ks_list)

    # ---------------------------
    # Dimensionality reduction
    # ---------------------------
    n_components = 3 if plot_3d else 2

    if method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=n_components, random_state=42)
    elif method == "isomap":
        from sklearn.manifold import Isomap

        reducer = Isomap(n_components=n_components, n_neighbors=10)
    elif method == "pca":
        from sklearn.decomposition import PCA

        pca_full = PCA()
        pca_full.fit(latent_vectors)
        explained_cum = np.cumsum(pca_full.explained_variance_ratio_)
        num_dims_99 = np.searchsorted(explained_cum, 0.99) + 1
        print(f"Number of dimensions needed to explain 99% variance: {num_dims_99}")
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f"Unknown method {method}")

    reduced = reducer.fit_transform(latent_vectors)

    # ---------------------------
    # Plotting
    # ---------------------------
    K_max = model.max_slots
    bounds = np.arange(0.5, K_max + 1.5, 1)
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=K_max)
    cmap = plt.get_cmap("tab10", K_max)

    fig = plt.figure(figsize=(8, 6))

    if plot_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            reduced[:, 2],
            c=true_ks_list,
            cmap=cmap,
            norm=norm,
            alpha=0.7,
        )
        ax.set_xlabel("Latent dim 1")
        ax.set_ylabel("Latent dim 2")
        ax.set_zlabel("Latent dim 3")
    else:
        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=true_ks_list,
            cmap=cmap,
            norm=norm,
            alpha=0.7,
        )
        plt.xlabel("Latent dim 1")
        plt.ylabel("Latent dim 2")

    cbar = plt.colorbar(scatter, ticks=range(1, K_max + 1))
    cbar.set_label("Number of components")
    plt.title(f"Latent space ({method.upper()}) — Key='{latent_key}' {mode_str}")
    plt.show()


@torch.no_grad()
def plot_slot_posteriors(
    model,
    dataset,
    idx,
    n_samples=100,
    device="cpu",
    use_gt_k=True,
    use_noise_encoder=False,
    freq_weight=3.0,
    freq_range=(2.5, 3.0),
):
    """
    Visualize posterior as a corner plot with slot-specific color coding.

    Args:
        model: trained SlotFlow
        dataset: torch Dataset object (must return (x_long, x_short, true_k, comps, params, noise_std))
        idx: index of the signal to sample from
        n_samples: number of posterior samples per slot
        device: "cpu" or "cuda"
        use_gt_k: whether to use ground truth K or model's prediction
        use_noise_encoder: whether the model uses a noise encoder
    """

    model.eval()
    model.to(device)

    # Unpack dataset sample
    item = dataset[idx]

    # Ensure consistent tuple length (pad with None if fewer elements)
    if len(item) == 6:
        x_long, x_short, true_k, comps, true_params, noise_std_true = item
        extra = None
        mode_str = "data generation in train mode"
    elif len(item) == 7:
        x_long, x_short, true_k, comps, true_params, noise_std_true, extra = item
        print("True parameters per component:")
        for i, (amp, phase, freq) in enumerate(true_params[:true_k]):
            print(
                f"  Slot {i}:  amplitude = {amp:.3f},  phase = {phase:.3f},  frequency = {freq:.3f}"
            )
        mode_str = "data generation in inference mode"
    else:
        raise ValueError(f"Unexpected number of outputs: {len(item)}")

    x_long = x_long.unsqueeze(0).to(device)
    x_short = x_short.unsqueeze(0).to(device)

    if use_gt_k:
        k_tensor = torch.tensor([true_k], device=device)
    else:
        k_tensor = None

    # Forward pass
    output = model(x_long, x_short, use_gt_k=k_tensor)
    context = output["context"]
    K = output["K_pred"].item()
    if len(item) == 7:
        print(f"Using {'GT' if use_gt_k else 'predicted'} K = {K}")

    slot_colors = plt.cm.tab10(np.linspace(0, 1, K))
    labels = ["amplitude", "phase", "frequency"]

    fig = None

    # --- Draw posterior samples per slot ---
    for k in range(K):
        slot_context = context[k].unsqueeze(0)

        param_samples = model.flow.sample(n_samples, context=slot_context)
        param_samples = param_samples.squeeze(0)

        amp = param_samples[:, 0].cpu().numpy()
        cos_phi = param_samples[:, 1]
        sin_phi = param_samples[:, 2]
        freq = (
            param_samples[:, 3].cpu().numpy() / freq_weight
            + (freq_range[0] + freq_range[1]) / 2
        )  # invert freq scaling

        # Reconstruct φ
        phase = torch.atan2(sin_phi, cos_phi).cpu().numpy() % (2 * np.pi)

        samples = np.stack([amp, phase, freq], axis=1)

        fig = corner.corner(
            samples,
            labels=labels,
            show_titles=False,
            color=slot_colors[k],
            fig=fig,
            hist_kwargs={"density": True},
            plot_contours=True,
            plot_datapoints=True,
            fill_contours=True,
        )

    # --- Overlay true params ---
    axes = np.array(fig.axes).reshape((len(labels), len(labels)))
    gt_colors = plt.cm.Dark2(np.linspace(0, 1, true_k))

    for param_idx, p in enumerate(true_params[:true_k]):
        amp, phi, freq = p.tolist()
        current_color = gt_colors[param_idx]
        for i in range(len(labels)):
            for j in range(i + 1):
                ax = axes[i, j]
                if i == j:
                    ax.axvline([amp, phi, freq][i], color=current_color, ls="--")
                else:
                    ax.axhline([amp, phi, freq][i], color=current_color, ls="--")
                    ax.axvline([amp, phi, freq][j], color=current_color, ls="--")
                    ax.plot(
                        [amp, phi, freq][j],
                        [amp, phi, freq][i],
                        "o",
                        color=current_color,
                    )

    plt.suptitle(
        f"Posterior over Parameters by Slot (Sample {idx}) – {mode_str}", fontsize=14
    )
    plt.tight_layout()
    plt.show()

    # --- Optional noise posterior ---
    if use_noise_encoder:
        mu = output["log_sigma_mu"][0].item()
        log_std = output["log_sigma_std"][0].item()
        sigma = np.exp(log_std)

        eps = np.random.randn(n_samples)
        log_sigma_samples = mu + sigma * eps
        sigma_samples = np.exp(log_sigma_samples)

        plt.figure()
        sns.kdeplot(
            sigma_samples, fill=True, color="cornflowerblue", alpha=0.5, label="KDE"
        )
        plt.axvline(
            noise_std_true, color="black", linestyle="--", linewidth=2, label="True σ"
        )
        plt.xlabel("σ")
        plt.ylabel("Posterior density")
        plt.title(f"Sampled Posterior over σ (Sample {idx})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


@torch.no_grad()
def plot_confusion_matrix(
    model,
    test_loader,
    device="cpu",
    num_samples=2000,
    normalize=False,
    cmap="Blues",
):
    """
    Evaluate the model's predictions and plot a confusion matrix using only matplotlib.

    Args:
        model: Trained SlotFlow model.
        test_loader: DataLoader with test dataset.
        device: Device to run inference on ("cpu" or "cuda").
        num_samples: Number of random samples to evaluate from the test set.
        normalize: If True, normalize rows of the confusion matrix.
        cmap: Colormap used for the matrix.

    Returns:
        accuracy (float): Overall prediction accuracy.
    """
    model.eval()
    model.to(device)

    dataset = test_loader.dataset
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    true_ks = []
    pred_ks = []

    # ---------------------------
    # Evaluate random subset
    # ---------------------------
    for idx in indices:
        item = dataset[idx]

        if len(item) == 6:
            x_long, x_short, true_k, _, _, _ = item
            mode_str = "(data generation in inference mode)"
        elif len(item) == 7:
            x_long, x_short, true_k, _, _, _, _ = item
            mode_str = "(data generation in train mode)"
        else:
            raise ValueError(f"Unexpected number of outputs: {len(item)}")

        x_long = x_long.unsqueeze(0).to(device)
        x_short = x_short.unsqueeze(0).to(device)

        output = model(x_long, x_short, use_gt_k=None)
        k_pred_int = output["K_pred"].item()

        true_ks.append(int(true_k))
        pred_ks.append(k_pred_int)

    # ---------------------------
    # Confusion matrix computation
    # ---------------------------
    labels = sorted(set(true_ks + pred_ks))
    label_to_index = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=np.int32)

    for t, p in zip(true_ks, pred_ks):
        i = label_to_index[t]
        j = label_to_index[p]
        cm[i, j] += 1

    if normalize:
        cm = cm.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        cm = cm / row_sums

    # ---------------------------
    # Accuracy
    # ---------------------------
    correct = sum(t == p for t, p in zip(true_ks, pred_ks))
    total = len(true_ks)
    accuracy = correct / total if total > 0 else 0

    # ---------------------------
    # Plotting
    # ---------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    ax.set_title(f"Confusion Matrix {mode_str} (Accuracy: {accuracy*100:.2f}%)")
    ax.set_xlabel("Predicted K")
    ax.set_ylabel("True K")
    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(n_labels):
        for j in range(n_labels):
            value = format(cm[i, j], fmt)
            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.show()

    return accuracy
