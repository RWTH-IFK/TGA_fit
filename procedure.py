import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel
from lmfit import Model

import fitting as ff


def read_TGA(tga_file, x_cutoff=None):
    tga_time_col = 'Time/min'
    tga_mass_col = 'Mass/%'
    tga_temp_col = 'Temp./°C'
    # Read TGA data
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            # Find the line number where the actual data starts and get header
            with open(tga_file, 'r', encoding=encoding) as f:
                header_row = None
                header_line = None
                for i, line in enumerate(f):
                    if line.startswith('##'):
                        header_line = line.strip('#').strip()
                        header_row = i
                        break

                if header_row is None:
                    continue

                print(f"Found header: {header_line}")  # Debug print

            # Get column names from the header line
            column_names = [col.strip() for col in header_line.split(';')]
            print(f"Column names: {column_names}")  # Debug print

            # Read the data with the header names
            tga_data = pd.read_csv(tga_file,
                                   sep=';',
                                   skiprows=header_row + 1,
                                   encoding=encoding,
                                   names=column_names)

            # Print available columns for debugging
            col_names = tga_data.columns.tolist()
            print(f"Available columns in TGA file: {col_names}")

            # Auto-detect columns if not specified
            if tga_time_col is None:
                time_candidates = [col for col in tga_data.columns if 'Time' in col]
                if time_candidates:
                    tga_time_col = time_candidates[0]
                    print(f"Auto-detected time column: {tga_time_col}")
                else:
                    raise ValueError("Could not auto-detect time column")

            if tga_mass_col is None:
                mass_candidates = [col for col in tga_data.columns if 'Mass' in col]
                if mass_candidates:
                    tga_mass_col = mass_candidates[0]
                    print(f"Auto-detected mass column: {tga_mass_col}")
                else:
                    raise ValueError("Could not auto-detect mass column")

            # Check if columns exist
            if tga_time_col not in tga_data.columns or tga_mass_col not in tga_data.columns:
                print(f"Required columns not found. Looking for {tga_time_col} and {tga_mass_col}")
                print(f"Available columns: {tga_data.columns.tolist()}")
                continue
            # Apply time cutoff if specified
            if x_cutoff is not None:
                tga_data = tga_data[tga_data[tga_time_col] <= x_cutoff]
            # Convert columns to numeric
            tga_data[tga_time_col] = pd.to_numeric(tga_data[tga_time_col], errors='coerce')
            tga_data[tga_mass_col] = pd.to_numeric(tga_data[tga_mass_col], errors='coerce')
            if tga_temp_col in col_names:
                tga_data[tga_temp_col] = pd.to_numeric(tga_data[tga_temp_col], errors='coerce')
            break

        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding}")
            continue
        except Exception as e:
            print(f"Attempt failed with encoding {encoding}: {e}")
            continue
    else:
        raise Exception("Could not read TGA file with any of the attempted encodings")
    y = tga_data[tga_mass_col]
    temperature = None
    time = None
    mass = None
    if tga_temp_col in col_names:
        temperature = tga_data[tga_temp_col].tolist()
    if tga_time_col in col_names:
        time = tga_data[tga_time_col].tolist()

    mass = tga_data[tga_mass_col].tolist()

    return mass, time, temperature


def num_of_procc(params):
    val = None
    num = 0
    for i in range(len(params)):
        try:
            val = params[f'scale_{i}']
            num += 1
        except KeyError:
            break
    return num


def plot_TGA_fit(result, TGA_dataset, stacking=False, ageinst_temp=False, save_csv=False):
    x = TGA_dataset.time
    if ageinst_temp:
        x = TGA_dataset.temp
    calc_TGA_procs = ff.calc_TGA(result.params, x, return_proccs=True)
    fit = sum(calc_TGA_procs)
    proc_num = len(calc_TGA_procs)
    fig, ax = plt.subplots()
    fig.set_size_inches(13, 9)
    ax.plot(x, TGA_dataset.y_data, marker='o', label='data')
    # 2nd Y axis with temperature
    if not ageinst_temp:
        color_T = 'green'
        ax_T = ax.twinx()
        ax_T.set_ylabel('T, C', color=color_T)
        line2 = ax_T.plot(TGA_dataset.time, TGA_dataset.temp,
                          color=color_T, label='T', linestyle='dotted')
        ax_T.tick_params(axis='y', labelcolor=color_T)

    if not stacking:
        for i in range(proc_num):
            label = f'proc_{i}'
            if i == 0:
                label = 'bkg'
                ax.plot(x, calc_TGA_procs[i], label=label, color='gray', ls='--')
            else:
                calc_TGA_procs[i] = calc_TGA_procs[i] + calc_TGA_procs[0]
                ax.plot(x, calc_TGA_procs[i], label=label)
    else:
        calc_TGA_procs_reversed = list(reversed(calc_TGA_procs))
        for i in range(proc_num):
            label = f'proc_{i}'
            if i == proc_num - 1:
                label = 'bkg'
                ax.plot(x, calc_TGA_procs_reversed[i], label=label, color='gray', ls='--')
                break
            if i == 0:
                calc_TGA_procs_reversed[i] = calc_TGA_procs_reversed[i] + calc_TGA_procs_reversed[
                    proc_num - 1]  # add bkg
            else:
                calc_TGA_procs_reversed[i] = calc_TGA_procs_reversed[i] + calc_TGA_procs_reversed[
                    i - 1]  # stack ontop of previos
            ax.fill_between(x, calc_TGA_procs_reversed[i], y2=calc_TGA_procs_reversed[proc_num - 1], label=label,
                            alpha=0.5)
    ax.plot(x, fit, color='red', label='fit', linewidth='3')
    ax.set_ylabel('mass, a.u.')
    ax.set_xlabel('t, min')
    if ageinst_temp:
        ax.set_xlabel('t, C')
    ax.legend(title='TGA fit')

    if save_csv:
        lines = list()
        result_path = 'result/TGA_fit.csv'
        # writing a header of needed length
        L_list = list()
        for i in range(len(calc_TGA_procs)):
            if i == 0:
                L_list.append(f'bkg')
            else:
                L_list.append(f'proc{i}')
        joined_L_list = ','.join(L_list)
        header = f't , T,  data, fit, {joined_L_list} \n'
        lines.append(header)
        for i, t in enumerate(TGA_dataset.time):
            procs = list()
            for j in range(len(calc_TGA_procs)):
                procs.append(f'{calc_TGA_procs[j][i]}')
            joined_procs = ','.join(procs)
            line = f'{str(t)}, {TGA_dataset.temp[i]},{TGA_dataset.y_data[i]}, {fit[i]}, {joined_procs} '
            line = line + '\n'
            lines.append(line)
        with open(result_path, 'w') as file:
            file.writelines(lines)

    fig.tight_layout()
    return fig


def save_report(report, path='result/report.txt'):
    with open(path, 'w') as file:
        file.writelines(report)


def plot_TEM_analysis(path, weight_area=False, remove_raws_from_end=4, distribution='both',
                      fix_amplitude=False, fit_method='fit', use_simple_gaussian=False):
    """
    Analyze particle size distribution from TEM data.
    Parameters:
        path (str): Path to the CSV file with TEM data
        weight_area (bool): Whether to include area-weighted distribution in the analysis
        remove_raws_from_end (int): Number of rows to remove from the end of the dataframe
        distribution (str): Type of distribution to fit - 'gaussian', 'lognormal', or 'both'
        fix_amplitude (bool): Whether to fix the amplitude to match the histogram maximum
        fit_method (str): Method to determine distribution parameters - 'fit' or 'stats'
                         'fit': Use curve fitting to determine parameters
                         'stats': Calculate parameters from data statistics
        use_simple_gaussian (bool): Whether to use a simple unnormalized Gaussian function
                                  instead of lmfit's GaussianModel
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import lmfit
    from lmfit.models import GaussianModel, LognormalModel
    from lmfit import Model
    from scipy import stats
    from scipy.optimize import curve_fit
    import math

    # Validate parameters
    if distribution not in ['gaussian', 'lognormal', 'both']:
        raise ValueError("distribution must be 'gaussian', 'lognormal', or 'both'")

    if fit_method not in ['fit', 'stats']:
        raise ValueError("fit_method must be 'fit' or 'stats'")

    # Load data
    file_path = path  # adjust as needed
    df = pd.read_csv(file_path, index_col=0)
    if remove_raws_from_end > 0:
        df = df.iloc[:-remove_raws_from_end]

    # Extract size column (assumed to be the last one)
    size_column = df.columns[-1]
    sizes = df[size_column].values  # Convert to numpy array

    # Compute weights as numpy arrays
    weights_number = np.ones_like(sizes)
    weights_area = sizes ** 2
    weights_volume = sizes ** 3

    # Define function for weighted mean and std deviation
    def weighted_stats(values, weights):
        average = np.average(values, weights=weights)
        variance = np.average((values - average) ** 2, weights=weights)
        std_dev = np.sqrt(variance)
        return average, std_dev

    # Calculate statistics
    mean_number, std_number = weighted_stats(sizes, weights_number)
    mean_area, std_area = weighted_stats(sizes, weights_area)
    mean_volume, std_volume = weighted_stats(sizes, weights_volume)

    # Print results
    print(f"Mean size (number-based): {mean_number:.2f} nm ± {std_number:.2f}")
    print(f"Mean size (area-weighted): {mean_area:.2f} nm ± {std_area:.2f}")
    print(f"Mean size (volume-weighted): {mean_volume:.2f} nm ± {std_volume:.2f}")

    def add_mean_line(mean, std, color):
        label = f'Mean = {mean:.1f} ± {std:.1f} nm'
        plt.axvline(mean, color=color, linestyle='--', label=label)

    # Define custom unnormalized Gaussian function
    def simple_gaussian(x, amplitude, mean, std):
        """Simple unnormalized Gaussian function"""
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    # Function to calculate chi-square and other goodness-of-fit metrics
    def calculate_goodness_of_fit(observed, expected):
        # Chi-square: sum((observed - expected)^2 / expected)
        # Only calculate for bins where expected > 0 to avoid division by zero
        valid_indices = expected > 0
        if sum(valid_indices) < 2:  # Need at least 2 points for meaningful calculation
            return float('nan'), float('nan'), float('nan'), float('nan')

        chi2 = np.sum(((observed[valid_indices] - expected[valid_indices]) ** 2) / expected[valid_indices])

        # Degrees of freedom: number of observations minus number of parameters (2 for Gaussian/Lognormal)
        dof = sum(valid_indices) - 2  # mean and std are the 2 parameters
        if dof <= 0:
            reduced_chi2 = float('nan')
        else:
            reduced_chi2 = chi2 / dof

        # AIC = 2k + n*ln(RSS/n) where k=number of parameters, n=sample size, RSS=residual sum of squares
        n = sum(valid_indices)
        RSS = np.sum((observed[valid_indices] - expected[valid_indices]) ** 2)
        AIC = 2 * 2 + n * np.log(RSS / n)  # 2 parameters

        # BIC = k*ln(n) + n*ln(RSS/n)
        BIC = 2 * np.log(n) + n * np.log(RSS / n)

        return chi2, reduced_chi2, AIC, BIC

    # Function to identify weight type
    def get_weight_type(weights):
        # Check if weights are all ones (number-weighted)
        if np.allclose(weights, np.ones_like(weights)):
            return "number"
        # Check if weights are proportional to size^3 (volume-weighted)
        elif np.allclose(weights / sizes ** 3, np.ones_like(weights) * (weights[0] / sizes[0] ** 3), rtol=1e-5):
            return "volume"
        # Check if weights are proportional to size^2 (area-weighted)
        elif np.allclose(weights / sizes ** 2, np.ones_like(weights) * (weights[0] / sizes[0] ** 2), rtol=1e-5):
            return "area"
        else:
            return "custom"

    # Function to fit and plot distributions using lmfit
    def analyze_and_plot_distributions(sizes, weights, x_min, x_max, initial_mean, initial_std,
                                       color='red', distribution_type='gaussian'):
        # Create histogram data
        counts, bin_edges = np.histogram(sizes, bins=n_bins, weights=weights)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find the max value of counts and its location
        max_count = counts.max()
        max_count_idx = np.argmax(counts)
        max_count_position = bin_centers[max_count_idx]

        # Determine weight type
        weight_type = get_weight_type(weights)

        # Initialize results dictionary
        fit_results = {}

        # Smooth x-axis for plotting
        x = np.linspace(x_min, x_max, 1000)

        # Process Gaussian if requested
        if distribution_type in ['gaussian', 'both']:
            if fit_method == 'stats':
                # Calculate Gaussian from statistics (like in the original code)
                gauss_mean = initial_mean
                gauss_std = initial_std

                # Calculate Gaussian using these parameters
                gauss_curve = max_count * np.exp(-((x - gauss_mean) ** 2) / (2 * gauss_std ** 2))

                gauss_label = f'Gaussian from Stats (μ={gauss_mean:.1f}, σ={gauss_std:.1f})'
                plt.plot(x, gauss_curve, color=color, linestyle='-', label=gauss_label)

                # Calculate expected counts at bin centers for chi-square calculation
                expected_gauss_counts = max_count * np.exp(-((bin_centers - gauss_mean) ** 2) / (2 * gauss_std ** 2))

                # Calculate goodness-of-fit metrics
                chi2, reduced_chi2, aic, bic = calculate_goodness_of_fit(counts, expected_gauss_counts)

                # Print chi-square statistics
                print(f"\nGaussian stats metrics for {weight_type}-weighted distribution:")
                print(f"Chi-square: {chi2:.3f}")
                print(f"Reduced chi-square: {reduced_chi2:.3f}")
                print(f"AIC: {aic:.3f}")
                print(f"BIC: {bic:.3f}")

                # Store results
                fit_results['gaussian'] = {
                    'mean': gauss_mean,
                    'std': gauss_std,
                    'method': 'stats',
                    'chi2': chi2,
                    'reduced_chi2': reduced_chi2,
                    'aic': aic,
                    'bic': bic
                }

            else:  # fit_method == 'fit'
                if use_simple_gaussian:
                    # Fit with simple unnormalized Gaussian function
                    try:
                        # Initial guess for the parameters
                        p0 = [max_count, max_count_position, initial_std]

                        if fix_amplitude:
                            # First fit to get shape parameters
                            popt, pcov = curve_fit(simple_gaussian, bin_centers, counts, p0=p0)

                            # Extract parameters
                            _, gauss_mean, gauss_std = popt

                            # Fix the mean and std, but let amplitude vary to match max height
                            def constrained_gaussian(x, amplitude):
                                return simple_gaussian(x, amplitude, gauss_mean, gauss_std)

                            # Re-fit with only amplitude as parameter
                            [amplitude], _ = curve_fit(constrained_gaussian, bin_centers, counts, p0=[max_count])

                            # Evaluate Gaussian at smooth x points
                            gauss_curve = simple_gaussian(x, amplitude, gauss_mean, gauss_std)

                            # Find maximum of the curve and scaling factor
                            fit_max = np.max(gauss_curve)
                            scale_factor = max_count / fit_max if fit_max > 0 else 1.0

                            # Scale curve to exactly match maximum
                            gauss_curve *= scale_factor
                            amplitude *= scale_factor

                        else:
                            # Fit all parameters
                            popt, pcov = curve_fit(simple_gaussian, bin_centers, counts, p0=p0)
                            amplitude, gauss_mean, gauss_std = popt

                            # Evaluate Gaussian at smooth x points
                            gauss_curve = simple_gaussian(x, amplitude, gauss_mean, gauss_std)
                            scale_factor = 1.0  # No scaling needed

                        # Plot the fitted Gaussian
                        gauss_label = f'Simple Gaussian Fit (μ={gauss_mean:.1f}, σ={gauss_std:.1f})'
                        plt.plot(x, gauss_curve, color=color, linestyle='-', label=gauss_label)

                        # Calculate expected counts for chi-square
                        expected_gauss_counts = simple_gaussian(bin_centers, amplitude, gauss_mean, gauss_std)
                        chi2, reduced_chi2, aic, bic = calculate_goodness_of_fit(counts, expected_gauss_counts)

                        # Store results
                        fit_results['gaussian'] = {
                            'mean': gauss_mean,
                            'std': gauss_std,
                            'amplitude': amplitude,
                            'method': 'simple_fit',
                            'chi2': chi2,
                            'reduced_chi2': reduced_chi2,
                            'aic': aic,
                            'bic': bic
                        }

                        # Print fit statistics
                        print(f"\nSimple Gaussian fit report for {weight_type}-weighted distribution:")
                        print(f"Chi-square: {chi2:.3f}")
                        print(f"Reduced chi-square: {reduced_chi2:.3f}")
                        print(f"AIC: {aic:.3f}")
                        print(f"BIC: {bic:.3f}")
                        if fix_amplitude:
                            print(f"Applied scaling factor: {scale_factor:.3f} to match maximum")

                    except Exception as e:
                        print(f"Error fitting Simple Gaussian: {e}")
                        # Fallback to stats method if fitting fails
                        gauss_mean = initial_mean
                        gauss_std = initial_std
                        gauss_curve = max_count * np.exp(-((x - gauss_mean) ** 2) / (2 * gauss_std ** 2))
                        plt.plot(x, gauss_curve, color=color, linestyle='-',
                                 label=f'Gaussian from Stats (fallback) (μ={gauss_mean:.1f}, σ={gauss_std:.1f})')

                        # Calculate expected counts for chi-square
                        expected_gauss_counts = max_count * np.exp(
                            -((bin_centers - gauss_mean) ** 2) / (2 * gauss_std ** 2))
                        chi2, reduced_chi2, aic, bic = calculate_goodness_of_fit(counts, expected_gauss_counts)

                        fit_results['gaussian'] = {
                            'mean': gauss_mean,
                            'std': gauss_std,
                            'method': 'stats_fallback',
                            'chi2': chi2,
                            'reduced_chi2': reduced_chi2,
                            'aic': aic,
                            'bic': bic
                        }
                else:
                    try:
                        # Create Gaussian model with lmfit
                        gauss_model = GaussianModel(prefix='gauss_')
                        gauss_params = gauss_model.make_params()

                        if fix_amplitude:
                            # First try with initial parameters to get the shape
                            gauss_params['gauss_amplitude'].set(max_count)
                            gauss_params['gauss_center'].set(
                                max_count_position)  # Use position of max as initial center
                            gauss_params['gauss_sigma'].set(initial_std)

                            # Fit model to get shape parameters
                            initial_fit = gauss_model.fit(counts, gauss_params, x=bin_centers)

                            # Now fix center and sigma, adjust amplitude to match max_count
                            gauss_center = initial_fit.best_values['gauss_center']
                            gauss_sigma = initial_fit.best_values['gauss_sigma']

                            # Create parameters with fixed center and sigma
                            gauss_params = gauss_model.make_params()
                            gauss_params['gauss_amplitude'].set(max_count)
                            gauss_params['gauss_center'].set(gauss_center, vary=False)
                            gauss_params['gauss_sigma'].set(gauss_sigma, vary=False)

                            # Re-fit with only amplitude varying
                            gauss_result = gauss_model.fit(counts, gauss_params, x=bin_centers)

                            # Evaluate model at x points
                            gauss_curve = gauss_model.eval(params=gauss_result.params, x=x)

                            # Find maximum of the curve and scaling factor
                            fit_max = np.max(gauss_curve)
                            scale_factor = max_count / fit_max if fit_max > 0 else 1.0

                            # Scale curve to exactly match maximum
                            gauss_curve *= scale_factor

                            # Get parameters for reporting
                            gauss_mean = gauss_result.best_values['gauss_center']
                            gauss_std = gauss_result.best_values['gauss_sigma']

                        else:
                            # If not fixing amplitude, perform regular fit
                            gauss_params['gauss_amplitude'].set(max_count)
                            gauss_params['gauss_center'].set(initial_mean)
                            gauss_params['gauss_sigma'].set(initial_std)

                            # Fit model
                            gauss_result = gauss_model.fit(counts, gauss_params, x=bin_centers)

                            # Get fitted parameters
                            gauss_mean = gauss_result.best_values['gauss_center']
                            gauss_std = gauss_result.best_values['gauss_sigma']

                            # Evaluate model at x points
                            gauss_curve = gauss_model.eval(params=gauss_result.params, x=x)
                            scale_factor = 1.0  # No scaling needed

                        # Plot the fitted Gaussian
                        gauss_label = f'Gaussian Fit (μ={gauss_mean:.1f}, σ={gauss_std:.1f})'
                        plt.plot(x, gauss_curve, color=color, linestyle='-', label=gauss_label)

                        # Store results
                        fit_results['gaussian'] = {
                            'mean': gauss_mean,
                            'std': gauss_std,
                            'result': gauss_result,
                            'aic': gauss_result.aic,
                            'bic': gauss_result.bic,
                            'method': 'fit'
                        }

                        # Print fit statistics
                        print(f"\nGaussian fit report for {weight_type}-weighted distribution:")
                        print(f"Reduced chi-square: {gauss_result.redchi:.3f}")
                        print(f"AIC: {gauss_result.aic:.3f}")
                        print(f"BIC: {gauss_result.bic:.3f}")
                        if fix_amplitude:
                            print(f"Applied scaling factor: {scale_factor:.3f} to match maximum")

                    except Exception as e:
                        print(f"Error fitting Gaussian: {e}")
                        # Fallback to stats method if fitting fails
                        gauss_mean = initial_mean
                        gauss_std = initial_std
                        gauss_curve = max_count * np.exp(-((x - gauss_mean) ** 2) / (2 * gauss_std ** 2))
                        plt.plot(x, gauss_curve, color=color, linestyle='-',
                                 label=f'Gaussian from Stats (fallback) (μ={gauss_mean:.1f}, σ={gauss_std:.1f})')

                        # Calculate expected counts for chi-square
                        expected_gauss_counts = max_count * np.exp(
                            -((bin_centers - gauss_mean) ** 2) / (2 * gauss_std ** 2))
                        chi2, reduced_chi2, aic, bic = calculate_goodness_of_fit(counts, expected_gauss_counts)

                        fit_results['gaussian'] = {
                            'mean': gauss_mean,
                            'std': gauss_std,
                            'method': 'stats_fallback',
                            'chi2': chi2,
                            'reduced_chi2': reduced_chi2,
                            'aic': aic,
                            'bic': bic
                        }

        # Process Lognormal if requested
        if distribution_type in ['lognormal', 'both']:
            if fit_method == 'stats':
                # Calculate lognormal parameters from statistics
                # For lognormal: if X is lognormal, then log(X) is normal
                # Need to calculate mu and sigma of the underlying normal distribution

                # Calculate shape and scale parameters from mean and std
                # Using the method of moments estimator
                phi = np.sqrt(1 + (initial_std / initial_mean) ** 2)
                lognorm_sigma = np.sqrt(np.log(phi ** 2))
                lognorm_center = np.log(initial_mean) - lognorm_sigma ** 2 / 2

                # Calculate mean and std for lognormal (in original units)
                lognorm_mean = np.exp(lognorm_center + lognorm_sigma ** 2 / 2)  # Should equal initial_mean
                lognorm_std = lognorm_mean * np.sqrt(np.exp(lognorm_sigma ** 2) - 1)  # Should equal initial_std

                # Calculate lognormal PDF
                # Remember that we need to scale it to match histogram height
                lognorm_curve = stats.lognorm.pdf(x, s=lognorm_sigma, scale=np.exp(lognorm_center))

                # Scale to match maximum height
                scale_factor = max_count / np.max(lognorm_curve) if np.max(lognorm_curve) > 0 else 1.0
                lognorm_curve *= scale_factor

                # Calculate expected counts at bin centers for chi-square
                expected_lognorm_counts = stats.lognorm.pdf(bin_centers, s=lognorm_sigma, scale=np.exp(lognorm_center))
                expected_lognorm_counts *= scale_factor

                # Calculate goodness-of-fit metrics
                chi2, reduced_chi2, aic, bic = calculate_goodness_of_fit(counts, expected_lognorm_counts)

                # Print chi-square statistics
                print(f"\nLognormal stats metrics for {weight_type}-weighted distribution:")
                print(f"Chi-square: {chi2:.3f}")
                print(f"Reduced chi-square: {reduced_chi2:.3f}")
                print(f"AIC: {aic:.3f}")
                print(f"BIC: {bic:.3f}")

                # Plot the lognormal curve
                lognorm_color = 'darkgreen' if distribution_type == 'both' else color
                lognorm_label = f'Lognormal from Stats (μ={lognorm_mean:.1f}, σ={lognorm_std:.1f})'
                plt.plot(x, lognorm_curve, color=lognorm_color, linestyle='--' if distribution_type == 'both' else '-',
                         label=lognorm_label)

                # Store results
                fit_results['lognormal'] = {
                    'mean': lognorm_mean,
                    'std': lognorm_std,
                    'center': lognorm_center,
                    'sigma': lognorm_sigma,
                    'method': 'stats',
                    'chi2': chi2,
                    'reduced_chi2': reduced_chi2,
                    'aic': aic,
                    'bic': bic
                }

            else:  # fit_method == 'fit'
                try:
                    # Create Lognormal model
                    lognorm_model = LognormalModel(prefix='lognorm_')
                    lognorm_params = lognorm_model.make_params()

                    if fix_amplitude:
                        # First try with initial parameters to get the shape
                        lognorm_params['lognorm_amplitude'].set(max_count)
                        lognorm_params['lognorm_center'].set(
                            np.log(max_count_position))  # Use position of max as initial center
                        lognorm_params['lognorm_sigma'].set(0.3)  # typical starting value

                        # Fit model to get shape parameters
                        initial_fit = lognorm_model.fit(counts, lognorm_params, x=bin_centers)

                        # Now fix center and sigma, adjust amplitude
                        lognorm_center = initial_fit.best_values['lognorm_center']
                        lognorm_sigma = initial_fit.best_values['lognorm_sigma']

                        # Create parameters with fixed center and sigma
                        lognorm_params = lognorm_model.make_params()
                        lognorm_params['lognorm_amplitude'].set(max_count)
                        lognorm_params['lognorm_center'].set(lognorm_center, vary=False)
                        lognorm_params['lognorm_sigma'].set(lognorm_sigma, vary=False)

                        # Re-fit with only amplitude varying
                        lognorm_result = lognorm_model.fit(counts, lognorm_params, x=bin_centers)

                        # Evaluate model at x points
                        lognorm_curve = lognorm_model.eval(params=lognorm_result.params, x=x)

                        # Find maximum of the curve and scaling factor
                        fit_max = np.max(lognorm_curve)
                        scale_factor = max_count / fit_max if fit_max > 0 else 1.0

                        # Scale curve to exactly match maximum
                        lognorm_curve *= scale_factor

                        # Get parameters for reporting
                        lognorm_center = lognorm_result.best_values['lognorm_center']
                        lognorm_sigma = lognorm_result.best_values['lognorm_sigma']

                    else:
                        # If not fixing amplitude, perform regular fit
                        lognorm_params['lognorm_amplitude'].set(max_count)
                        lognorm_params['lognorm_center'].set(np.log(initial_mean))
                        lognorm_params['lognorm_sigma'].set(0.3)

                        # Fit model
                        lognorm_result = lognorm_model.fit(counts, lognorm_params, x=bin_centers)

                        # Get fitted parameters
                        lognorm_center = lognorm_result.best_values['lognorm_center']
                        lognorm_sigma = lognorm_result.best_values['lognorm_sigma']

                        # Evaluate model at x points
                        lognorm_curve = lognorm_model.eval(params=lognorm_result.params, x=x)
                        scale_factor = 1.0  # No scaling needed

                    # Calculate mean and standard deviation for lognormal (in original units)
                    lognorm_mean = np.exp(lognorm_center + lognorm_sigma ** 2 / 2)
                    lognorm_std = lognorm_mean * np.sqrt(np.exp(lognorm_sigma ** 2) - 1)

                    # Plot the fitted Lognormal
                    lognorm_color = 'darkgreen' if distribution_type == 'both' else color
                    lognorm_label = f'Lognormal Fit (μ={lognorm_mean:.1f}, σ={lognorm_std:.1f})'
                    plt.plot(x, lognorm_curve, color=lognorm_color,
                             linestyle='--' if distribution_type == 'both' else '-',
                             label=lognorm_label)

                    # Store results
                    fit_results['lognormal'] = {
                        'mean': lognorm_mean,
                        'std': lognorm_std,
                        'center': lognorm_center,
                        'sigma': lognorm_sigma,
                        'result': lognorm_result,
                        'aic': lognorm_result.aic,
                        'bic': lognorm_result.bic,
                        'method': 'fit'
                    }

                    # Print fit statistics
                    print(f"\nLognormal fit report for {weight_type}-weighted distribution:")
                    print(f"Reduced chi-square: {lognorm_result.redchi:.3f}")
                    print(f"AIC: {lognorm_result.aic:.3f}")
                    print(f"BIC: {lognorm_result.bic:.3f}")
                    if fix_amplitude:
                        print(f"Applied scaling factor: {scale_factor:.3f} to match maximum")

                except Exception as e:
                    print(f"Error fitting Lognormal: {e}")
                    # Fallback to stats method if fitting fails
                    phi = np.sqrt(1 + (initial_std / initial_mean) ** 2)
                    lognorm_sigma = np.sqrt(np.log(phi ** 2))
                    lognorm_center = np.log(initial_mean) - lognorm_sigma ** 2 / 2

                    lognorm_mean = np.exp(lognorm_center + lognorm_sigma ** 2 / 2)
                    lognorm_std = lognorm_mean * np.sqrt(np.exp(lognorm_sigma ** 2) - 1)

                    lognorm_curve = stats.lognorm.pdf(x, s=lognorm_sigma, scale=np.exp(lognorm_center))
                    scale_factor = max_count / np.max(lognorm_curve) if np.max(lognorm_curve) > 0 else 1.0
                    lognorm_curve *= scale_factor

                    # Calculate expected counts for chi-square
                    expected_lognorm_counts = stats.lognorm.pdf(bin_centers, s=lognorm_sigma,
                                                                scale=np.exp(lognorm_center))
                    expected_lognorm_counts *= scale_factor
                    chi2, reduced_chi2, aic, bic = calculate_goodness_of_fit(counts, expected_lognorm_counts)

                    lognorm_color = 'darkgreen' if distribution_type == 'both' else color
                    plt.plot(x, lognorm_curve, color=lognorm_color,
                             linestyle='--' if distribution_type == 'both' else '-',
                             label=f'Lognormal from Stats (fallback) (μ={lognorm_mean:.1f}, σ={lognorm_std:.1f})')

                    fit_results['lognormal'] = {
                        'mean': lognorm_mean,
                        'std': lognorm_std,
                        'center': lognorm_center,
                        'sigma': lognorm_sigma,
                        'method': 'stats_fallback',
                        'chi2': chi2,
                        'reduced_chi2': reduced_chi2,
                        'aic': aic,
                        'bic': bic
                    }

        # Determine best distribution if both were fitted/calculated
        if distribution_type == 'both' and 'gaussian' in fit_results and 'lognormal' in fit_results:
            # Compare metrics to determine best model
            if fit_method == 'fit':
                if 'aic' in fit_results['gaussian'] and 'aic' in fit_results['lognormal']:
                    gauss_aic = fit_results['gaussian']['aic']
                    lognorm_aic = fit_results['lognormal']['aic']
                    best_model = 'gaussian' if gauss_aic < lognorm_aic else 'lognormal'
                    print(
                        f"Best fit model based on AIC: {best_model.capitalize()} (AIC: {min(gauss_aic, lognorm_aic):.3f})")
            else:  # fit_method == 'stats'
                if 'aic' in fit_results['gaussian'] and 'aic' in fit_results['lognormal']:
                    gauss_aic = fit_results['gaussian']['aic']
                    lognorm_aic = fit_results['lognormal']['aic']
                    best_model = 'gaussian' if gauss_aic < lognorm_aic else 'lognormal'
                    print(
                        f"Best model based on AIC: {best_model.capitalize()} (AIC: {min(gauss_aic, lognorm_aic):.3f})")

                if 'reduced_chi2' in fit_results['gaussian'] and 'reduced_chi2' in fit_results['lognormal']:
                    gauss_chi2 = fit_results['gaussian']['reduced_chi2']
                    lognorm_chi2 = fit_results['lognormal']['reduced_chi2']
                    best_model_chi2 = 'gaussian' if gauss_chi2 < lognorm_chi2 else 'lognormal'
                    print(
                        f"Best model based on reduced chi-square: {best_model_chi2.capitalize()} (χ²: {min(gauss_chi2, lognorm_chi2):.3f})")

            # Use the mean and std from best model if determined
            if 'aic' in fit_results['gaussian'] and 'aic' in fit_results['lognormal']:
                if best_model == 'gaussian':
                    mean_fit = fit_results['gaussian']['mean']
                    std_fit = fit_results['gaussian']['std']
                else:
                    mean_fit = fit_results['lognormal']['mean']
                    std_fit = fit_results['lognormal']['std']
            else:
                # Default to Gaussian stats
                mean_fit = fit_results['gaussian']['mean']
                std_fit = fit_results['gaussian']['std']
        elif 'gaussian' in fit_results:
            mean_fit = fit_results['gaussian']['mean']
            std_fit = fit_results['gaussian']['std']
        elif 'lognormal' in fit_results:
            mean_fit = fit_results['lognormal']['mean']
            std_fit = fit_results['lognormal']['std']
        else:
            mean_fit = initial_mean
            std_fit = initial_std

        return mean_fit, std_fit, fit_results

    # Plot histograms
    plt.figure(figsize=(15, 5))
    columns = 2
    n_bins = int(np.ceil(np.sqrt(len(sizes))))
    x_min = 2
    x_max = 17

    if weight_area:
        columns = 3

    # Number-weighted
    plt.subplot(1, columns, 1)
    plt.hist(sizes, bins=n_bins, weights=weights_number, color='skyblue', edgecolor='black')
    plt.xlim(x_min, x_max)
    add_mean_line(mean_number, std_number, 'blue')

    # Analyze and plot distributions for number-weighted
    mean_fit_number, std_fit_number, result_number = analyze_and_plot_distributions(
        sizes, weights_number, x_min, x_max, mean_number, std_number, 'red', distribution)

    plt.title("Number-weighted Distribution")
    plt.xlabel("Size (nm)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    # Area-weighted
    if weight_area:
        plt.subplot(1, columns, 2)
        plt.hist(sizes, bins=n_bins, weights=weights_area, color='lightgreen', edgecolor='black')
        plt.xlim(x_min, x_max)
        add_mean_line(mean_area, std_area, 'green')

        # Analyze and plot distributions for area-weighted
        mean_fit_area, std_fit_area, result_area = analyze_and_plot_distributions(
            sizes, weights_area, x_min, x_max, mean_area, std_area, 'darkgreen', distribution)

        plt.title("Surface Area-weighted Distribution")
        plt.xlabel("Size (nm)")
        plt.ylabel("Weighted Sum (∝ size²)")
        plt.legend()
        plt.grid(True)

    # Volume-weighted
    plt.subplot(1, columns, columns if weight_area else 2)
    plt.hist(sizes, bins=n_bins, weights=weights_volume, color='salmon', edgecolor='black')
    plt.xlim(x_min, x_max)
    add_mean_line(mean_volume, std_volume, 'red')

    # Analyze and plot distributions for volume-weighted
    mean_fit_volume, std_fit_volume, result_volume = analyze_and_plot_distributions(
        sizes, weights_volume, x_min, x_max, mean_volume, std_volume, 'darkred', distribution)

    plt.title("Volume-weighted Distribution")
    plt.xlabel("Size (nm)")
    plt.ylabel("Weighted Sum (∝ size³)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print fitted results
    print("\nSummary of Fitted parameters:")
    print(f"Number-weighted: Mean = {mean_fit_number:.2f} nm, Std = {std_fit_number:.2f}")
    if weight_area:
        print(f"Area-weighted: Mean = {mean_fit_area:.2f} nm, Std = {std_fit_area:.2f}")
    print(f"Volume-weighted: Mean = {mean_fit_volume:.2f} nm, Std = {std_fit_volume:.2f}")

    # Return the fit results for potential further analysis
    if weight_area:
        return result_number, result_area, result_volume
    else:
        return result_number, result_volume
