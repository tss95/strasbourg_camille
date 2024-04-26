import os
from src.Transforms import ResampleTrace, AddNoise, BandpassFilter, AverageInterval, Taper, LocalMinMaxNorm, FixedRangeNormalize, ApplyRandomColormap, GAFTransform
from torchvision import transforms


def setup_trace_transforms(fs, 
                           decimate=False, 
                           add_noise=False, 
                           bandpass_first=False, 
                           average_interval=False, 
                           taper=False, 
                           minmax=False, 
                           use_gaf = False, 
                           apply_colormap = False,
                           decimation_factor=4, 
                           noise_level=0.05, 
                           lowcut=0.1, 
                           highcut=20, 
                           order=4, 
                           interval_max_length=100,
                           taper_max_percentage=0.05,
                           colormaps=['viridis', 'plasma', 'inferno', 'magma', 'cividis']):
    transformations = []
    updated_fs = fs
    
    # Decimation
    if decimate:
        transformations.append(ResampleTrace(decimation_factor))
        updated_fs /= decimation_factor

    # Add Noise
    if add_noise:
        transformations.append(AddNoise(noise_level))

    # Bandpass Filter
    if bandpass_first:
        transformations.append(BandpassFilter(lowcut, highcut, updated_fs, order))

    # Average Interval
    if average_interval:
        transformations.append(AverageInterval(interval_max_length))

    # Taper
    if taper:
        transformations.append(Taper(taper_max_percentage))

    # Local Min-Max Normalization
    if minmax:
        transformations.append(LocalMinMaxNorm())
        
    if use_gaf:
        transformations.append(GAFTransform())
    
    if apply_colormap:
        transformations.append(ApplyRandomColormap(colormaps))

    return transforms.Compose(transformations)

def setup_mfr_transforms(apply_colormap=True, colormaps=['viridis', 'plasma', 'inferno', 'magma', 'cividis']):
    transformations = []
    transformations.append(FixedRangeNormalize())  # Normalize based on fixed range
    if apply_colormap:
        transformations.append(ApplyRandomColormap(colormaps))

    return transforms.Compose(transformations)

def setup_output_dirs(run_id, cfg, make_folders = True):
    root_dir = cfg.data_paths.project_path
    base_dir = os.path.join(root_dir, cfg.output_paths.storage_dir, cfg.model_name)
    model_plot_folder = os.path.join(base_dir, run_id, cfg.output_paths.plots_folder)
    model_weights_folder = os.path.join(base_dir, run_id, cfg.output_paths.model_weights_folder)
    #pretrained_model_path = os.path.join(base_dir, cfg.pretrained_model_path)
    if make_folders:
        os.makedirs(model_plot_folder, exist_ok=True)
        os.makedirs(model_weights_folder, exist_ok=True)
    cfg.output_paths.plots_folder = model_plot_folder
    cfg.output_paths.model_weights_folder = model_weights_folder
    #cfg.pretrained_model_path = pretrained_model_path
    return cfg