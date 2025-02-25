import csv

import pandas as pd
from matplotlib import pyplot as plt

import fitting as ff


def read_TGA(tga_file, x_cutoff = None):
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


def plot_TGA_fit(result, TGA_dataset, stacking = False, ageinst_temp = False, save_csv = False):
    x = TGA_dataset.time
    if ageinst_temp:
        x = TGA_dataset.temp
    calc_TGA_procs = ff.calc_TGA(result.params, x, return_proccs=True)
    fit = sum(calc_TGA_procs)
    proc_num = len(calc_TGA_procs)
    fig, ax = plt.subplots()
    fig.set_size_inches(13, 9)
    ax.plot(x, TGA_dataset.y_data, marker='o', label = 'data')
    # 2nd Y axis with temperature
    if not ageinst_temp:
        color_T = 'green'
        ax_T = ax.twinx()
        ax_T.set_ylabel('T, K', color=color_T)
        line2 = ax_T.plot(TGA_dataset.time, TGA_dataset.temp,
                         color=color_T, label='T', linestyle = 'dotted')
        ax_T.tick_params(axis='y', labelcolor=color_T)

    if not stacking:
        for i in range(proc_num):
            label = f'proc_{i}'
            if i == 0:
                label = 'bkg'
                ax.plot(x, calc_TGA_procs[i], label=label, color = 'gray', ls='--')
            else:
                calc_TGA_procs[i] = calc_TGA_procs[i] + calc_TGA_procs[0]
                ax.plot(x, calc_TGA_procs[i], label=label)
    else:
        calc_TGA_procs_reversed = list(reversed(calc_TGA_procs))
        for i in range(proc_num):
            label = f'proc_{i}'
            if i == proc_num-1:
                label = 'bkg'
                ax.plot(x, calc_TGA_procs_reversed[i], label=label, color='gray', ls='--')
                break
            if i == 0:
                calc_TGA_procs_reversed[i] = calc_TGA_procs_reversed[i] + calc_TGA_procs_reversed[proc_num - 1] # add bkg
            else:
                calc_TGA_procs_reversed[i] = calc_TGA_procs_reversed[i] + calc_TGA_procs_reversed[i-1] # stack ontop of previos
            ax.fill_between(x, calc_TGA_procs_reversed[i],y2= calc_TGA_procs_reversed[proc_num - 1], label=label,alpha=0.5 )
    ax.plot(x, fit, color='red', label = 'fit', linewidth = '3')
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

def save_report(report, path = 'result/report.txt'):
    with open(path, 'w') as file:
        file.writelines(report)

