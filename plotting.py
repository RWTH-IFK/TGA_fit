import pandas as pd
import matplotlib.pyplot as plt


def plot_tga_ms(tga_file, ms_file,
                tga_time_col='Time',
                tga_mass_col='Mass',
                ms_time_col='Elapsed Time (s)',
                ms_co2_col='Carbon dioxide',
                time_cutoff=None,
                plot_temperature=True,
                figsize=(10, 6)
                ):
    """
    Create a plot combining TGA mass loss and MS carbon dioxide data.

    Parameters:
    -----------
    tga_file : str
        Path to the TGA text file
    ms_file : str
        Path to the MS CSV file
    tga_time_col : str
        Name of the time column in TGA file
    tga_mass_col : str
        Name of the mass column in TGA file
    ms_time_col : str
        Name of the time column in MS file
    ms_co2_col : str
        Name of the CO2 column in MS file
    time_cutoff : float or None
        Maximum time in minutes to plot. If None, all data is plotted
    figsize : tuple
        Figure size in inches (width, height)

    Returns:
    --------
    fig, (ax1, ax2) : tuple
        Figure and axes objects for further customization if needed
    """
    plt.rcParams.update({'font.size': 18})
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
            print(f"Available columns in TGA file: {tga_data.columns.tolist()}")

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

            # Convert columns to numeric
            tga_data[tga_time_col] = pd.to_numeric(tga_data[tga_time_col], errors='coerce')
            tga_data[tga_mass_col] = pd.to_numeric(tga_data[tga_mass_col], errors='coerce')
            tga_data['Temp./°C'] = pd.to_numeric(tga_data['Temp./°C'], errors='coerce')
            break

        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding}")
            continue
        except Exception as e:
            print(f"Attempt failed with encoding {encoding}: {e}")
            continue
    else:
        raise Exception("Could not read TGA file with any of the attempted encodings")

    # Read MS data
    try:
        ms_data = pd.read_csv(ms_file)

        # Check if required columns exist
        if ms_time_col not in ms_data.columns or ms_co2_col not in ms_data.columns:
            print("Available columns in MS file:", ms_data.columns.tolist())
            raise ValueError(f"Required columns not found in MS file")

        # Convert time column to numeric and then to minutes
        ms_data[ms_time_col] = pd.to_numeric(ms_data[ms_time_col], errors='coerce')
        ms_data[ms_co2_col] = pd.to_numeric(ms_data[ms_co2_col], errors='coerce')
        ms_data['Time_min'] = ms_data[ms_time_col] / 60
    except Exception as e:
        raise Exception(f"Error reading MS file: {e}")

    # Remove any NaN values
    tga_data = tga_data.dropna(subset=[tga_time_col, tga_mass_col])
    ms_data = ms_data.dropna(subset=[ms_time_col, ms_co2_col])

    # Apply time cutoff if specified
    if time_cutoff is not None:
        tga_data = tga_data[tga_data[tga_time_col] <= time_cutoff]
        ms_data = ms_data[ms_data['Time_min'] <= time_cutoff]

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot Mass data on primary y-axis
    color1 = 'blue'
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Mass (%)', color=color1)
    line1 = ax1.plot(tga_data[tga_time_col], tga_data[tga_mass_col],
                     color=color1, label='Mass')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Create secondary y-axis and plot Carbon dioxide data
    color2 = 'green'
    ax2 = ax1.twinx()
    color2 = 'red'
    ax2.set_ylabel('Carbon dioxide (bar)', color=color2)
    line2 = ax2.plot(ms_data['Time_min'], ms_data[ms_co2_col],
                     color=color2, label='CO2')
    ax2.tick_params(axis='y', labelcolor=color2)

    if plot_temperature:
        color3 = 'green'
        # Plot Temperature data on right y-axis
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.1))
        color2 = 'red'
        ax3.set_ylabel('Temp (C)', color=color3)
        line2 = ax3.plot(tga_data[tga_time_col], tga_data['Temp./°C'],
                         color=color3, label='Temp, C')
        ax3.tick_params(axis='y', labelcolor=color3)

    # Add title and adjust layout
    plt.title('TGA Mass Loss and Carbon Dioxide Evolution')

    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    fig.tight_layout()

    return fig, (ax1, ax2, ax3)