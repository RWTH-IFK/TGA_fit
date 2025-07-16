import numpy as np
import lmfit as lmf
from plotting import plot_tga_ms as plot
import matplotlib.pyplot as plt
import fitting as ff
import procedure as prd
import calculations as calc
from uncertainties import ufloat
import simulation as sim
from ase import io


def do_fit(path, x_cutoff=161, end_weight=0.67, n_proc=6, x_temp=False, init_guess=None):
    return_residual = True
    TGA_dataset = ff.TGA_dataset()
    TGA_dataset.y_data, TGA_dataset.time, TGA_dataset.temp = prd.read_TGA(path, x_cutoff=x_cutoff)
    end_weight = end_weight

    # zoom
    # TGA_dataset.y_data, TGA_dataset.time, TGA_dataset.temp = prd.read_TGA(path, x_cutoff=50)
    # end_weight = 0.9153

    x_axis = TGA_dataset.time
    if x_temp:
        x_axis = TGA_dataset.temp

    params = lmf.Parameters()
    if init_guess is None:
        params = ff.create_params(n_proc, end_weight, x_axis)
    else:
        # count params
        process_indices = set()
        for param_name in init_guess.keys():
            if param_name in ['bkg']:  # Skip non-process parameters
                continue
            if '_' in param_name:
                try:
                    index = int(param_name.split('_')[1])
                    process_indices.add(index)
                except (ValueError, IndexError):
                    continue
        n_proc = len(process_indices)
        print(f'{n_proc} parameters found in initial guess')
        params = ff.create_parameters_from_dict(init_guess)
    TGA_dataset.y_data[:] = [x / 100 for x in TGA_dataset.y_data]

    num_iterations = 8000
    result = lmf.minimize(ff.objective, params,
                          args=(
                              # pass to objective
                              TGA_dataset, return_residual, x_temp
                          ),
                          nan_policy='omit', max_nfev=int(num_iterations),  # **fit_kws ampgo
                          )

    # plotting

    report = lmf.fit_report(result.params)
    print(report)
    print(result.message)
    plt.rcParams.update({'font.size': 20})
    fig = prd.plot_TGA_fit(result, TGA_dataset, stacking=False, ageinst_temp=x_temp, save_csv=True)
    fig.savefig('result/result_fit.png')
    plt.show()
    prd.save_report(report)


def plotting():
    path_TGA = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Citrate_samples\water layers\TGA/IONP_25.09.txt"
    path_MS = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Citrate_samples\water layers\TGA/IONP_25.09.csv"

    fig, (ax1, ax2, ax3) = plot(path_TGA, path_MS,
                                tga_time_col='Time/min',  # This matches your actual column name
                                tga_mass_col='Mass/%',  # This matches your actual column name
                                ms_time_col='Elapsed Time (s)',
                                ms_co2_col='Carbon dioxide',
                                time_cutoff=200,
                                plot_temperature=True
                                )
    plt.show()


if __name__ == '__main__':
    path_diffRH = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Citrate_samples\water layers\TGA/IONP_25.09_75RH.txt"
    path = r'C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Washed_citrate\TGA\compare_ramp\5K_min_T.txt'

    path = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\TGA_ligand_count\IONP_25.09\Data\IONP_25.09_75RH.txt"
    path_deg30 = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\TGA_ligand_count\MP_20.02.25-2\ExpDat_TA_00637_30RH.txt"
    path_deg75 = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\TGA_ligand_count\MP_20.02.25-2\ExpDat_TA_00716_75_RH.txt"
    # do_fit(path_deg30, x_cutoff=40, n_proc=5, end_weight=0.966, x_temp = False) # okayish DEG30
    # do_fit(path_deg75, x_cutoff=43, n_proc=3, end_weight=0.9131, x_temp = False) # Okayish DEG75
    # do_fit(path_deg30, x_cutoff=190, n_proc=5, end_weight=0.82, x_temp=False) #full DEG30
    # do_fit(path_deg75, x_cutoff=177, n_proc=4, end_weight=0.825, x_temp=False)  # full DEG75
    # do_fit(path_deg75, x_cutoff=42, n_proc=5, end_weight=0.9121, x_temp=False)
    # do_fit(path, x_cutoff=44, n_proc=3, end_weight=0.8791, x_temp = True)
    # do_fit(path, x_cutoff=44, n_proc=3, end_weight=0.8791, x_temp=False)
    # plotting()

    # with guess
    # Tuple format: (value, vary, min, max)
    #DEG 75
    # initial_guesses = {
    #     'bkg': 0.9131,  # Will be automatically fixed
    #     'scale_0': (0.05207982, True, 0.0, 1000),
    #     'm_0': (15.1580796, True, 0.0, 1000),
    #     'G_0': (0.5277550, True, 0.0, 1000),
    #     'scale_1': (4.2787e-09, True, 0.0, 1000),
    #     'm_1': (1.9059842, False, 0.0, 1000),
    #     'G_1': (0.03325078, False, 0.0, 1000),  # (value, vary, min, max)
    #     'scale_2': (0.0372733, True, 0.0, 1000),
    #     'm_2': (2.1172730, False, 0.0, 1000),
    #     'G_2': (0.0784263, False, 0.0, 1000),  # (value, vary, min, max)
    #
    # }
    best_guess = {
        'bkg': 0.9087,  # Will be automatically fixed
        'scale_0': (0.06929625, True, 0.0, 1000),
        'm_0': (14.2659438, True, 0.0, 1000),
        'G_0': (0.40380599, True, 0.0, 1000),
        'scale_1': (4.2787e-09, False, 0.0, 4.2787e-08),
        'm_1': (1.9059842, False, 0.0, 1000),
        'G_1': (0.03325078, False, 0.0, 1000),  # (value, vary, min, max)
        'scale_2': (0.019, True, 0.0, 1000),
        'm_2': (32, True, 32.0,1000),
        'G_2': (0.1, True, 0.005, 1000),  # (value, vary, min, max)

    }
    initial_guesses = {
        'bkg': 0.9131,  # Will be automatically fixed
        'scale_0': (0.05207982, True, 0.0, 1000),
        'm_0': (15.1580796, True, 0.0, 1000),
        'G_0': (0.5277550, True, 0.0, 1000),
        'scale_1': (4.2787e-09, True, 0.0, 1000),
        'm_1': (1.9059842, False, 0.0, 1000),
        'G_1': (0.03325078, False, 0.0, 1000),  # (value, vary, min, max)
        'scale_2': (0.0372733, True, 0.0, 1000),
        'm_2': (2.1172730, False, 0.0, 1000),
        'G_2': (0.0784263, False, 0.0, 1000),  # (value, vary, min, max)

    }
    continued_guess = {
        'bkg': 0.8956,  # Will be automatically fixed
        'scale_0': (0.06929625, True, 0.0, 1000),
        'm_0': (14.2659438, True, 0.0, 1000),
        'G_0': (0.40380599, True, 0.0, 1000),
        'scale_1': (0.01138252, False, 0.0, 1000),
        'm_1': (48.5964929, False, 0.0, 1000),
        'G_1': (0.84100243, False, 0.0, 1000),  # (value, vary, min, max)
        'scale_2': (0.020, True, 0.0, 1000),
        'm_2': (34, True, 32.0,40),
        'G_2': (0.1, True, 0.005, 1000),  # (value, vary, min, max)

    }
    # do_fit(path_deg75, x_cutoff=43, n_proc=3, end_weight=0.9131, x_temp=False, init_guess=initial_guesses) #okay fit
    # do_fit(path_deg75, x_cutoff=43, n_proc=4, end_weight=0.8398, x_temp=False, init_guess=best_guess) #best fit
    # do_fit(path_deg75, x_cutoff=51, n_proc=4, end_weight=0.8398, x_temp=False, init_guess=continued_guess)
    #DEG30
    initial_guesses = {
        'bkg': 0.95605,  # Will be automatically fixed
        'scale_0': (0.01607593, True, 0.0, 1000),
        'm_0': (24.6364885, True, 0.0, 1000),
        'G_0': (0.13, True, 0.0, 1000),
        'scale_1': (0.5, True, 0.0, 1000),
        'm_1': (1.9059842, True, 0.0, 1000),
        'G_1': (0.03325078, True, 0.0, 1000),  # (value, vary, min, max)
        'scale_2': (0.01575051, True, 0.0, 1000),
        'm_2': (10.2664391, True, 0.0, 1000),
        'G_2': (0.34286499, True, 0.0, 1000),  # (value, vary, min, max)

    }
    # do_fit(path_deg30, x_cutoff=48, n_proc=3, end_weight=0.79, x_temp=False, init_guess=initial_guesses)
    continued_guesses = {
        'bkg': 0.9678,  # Will be automatically fixed
        'scale_0': (0.00497583, True, 0.00397583, 0.00597583),
        'm_0': (12.5489043, True, 0.0, 1000),
        'G_0': (0.74424954, True, 0.0, 1000),
        'scale_1': (0.01190142, True, 0.0, 1000),
        'm_1': (43.8715867, True, 30.0, 1000),
        'G_1': (0.52661112, True, 0.0, 1000),  # (value, vary, min, max)
        # 'scale_2': (0.010, True, 0.0, 1000),
        # 'm_2': (34, True, 32.0,45),
        # 'G_2': (0.1, True, 0.005, 1000),  # (value, vary, min, max)

    }
    do_fit(path_deg30, x_cutoff=40, n_proc=3, end_weight=0.79, x_temp=False, init_guess=continued_guesses)

    path_30RH = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Citrate_samples\water layers\TGA\IONP_25.09.txt"
    path_75RH = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Citrate_samples\water layers\TGA\IONP_25.09_75RH.txt"
    # do_fit(path_30RH, x_cutoff=40, n_proc=3, end_weight=0.94, x_temp=False) #good SB procces
    # do_fit(path_30RH, x_cutoff=46, n_proc=3, end_weight=0.944, x_temp=True)
    # do_fit(path_30RH, x_cutoff=36, n_proc=3, end_weight=0.94, x_temp=False)
    # do_fit(path_30RH, x_cutoff=39, n_proc=3, end_weight=0.9473, x_temp=False)
    # do_fit(path_75RH, x_cutoff=239, n_proc=6, end_weight=0.6473, x_temp=False)

    path_deg = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\TGA_ligand_count\MP_20.02.25-2\ExpDat_TA_00637.txt"
    # do_fit(path_deg, x_cutoff=175, n_proc=4, end_weight=0.8619, x_temp=False)
    # do_fit(path_deg, x_cutoff=80, n_proc=4, end_weight=0.8872, x_temp=True)

    # NP = calc.Nanoparticle(ligand_type='DEG', composition_NP='Fe3O4', wt_NP=ufloat(86.19, 0.5),
    #                        wt_water=ufloat(5.2, 0.14), diameter_NP=ufloat(8.1, 1.4))
    # New DEG
    # NP = calc.Nanoparticle(ligand_type='DEG', composition_NP='Fe3O4', wt_NP=ufloat(62, 0.5),
    #                        wt_water=ufloat(7.5, 0.27), diameter_NP=ufloat(4.2, 0.7))
    # NP.calculate_sphere()
    # NP.calculate_trunc_octahedron()
    # NP.generate_report(save=False)

    # sim.tryout()
    # MD.try_lammps()
    path_cif = r"C:\Users\admin-lap143\RWTH\Students\Evgeny\IONP\XRD\magnetite.cif"
    # path_cif = r"C:\Users\admin-lap143\Downloads\Telegram Desktop\Fe3O4.cif"
    # a = sim.IONPs_surface(2, binding_indices=[19,18], ligand_offset=1, to_view = ['model'], optimize_rotation = True, miller=(0,0,1), tol = 0.5, path = path_cif)
    # a = sim.IONPs_surface(5, binding_indices=[19], ligand_offset=1, to_view=['model'], optimize_rotation = True, miller=(1,1,1),path = path_cif)
    # a = sim.IONPs_surface(2.0, binding_indices=[18, 19, 20], ligand_offset=1.4, to_view=['model'],
    #                       optimize_rotation=True, alligne=False)
    # io.write('result_4.xyz', a, format='xyz')

    # TEM
    # path_TEM_20_02_25 = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\TEM\24_06_25\24_06_25\IONP 20_02_25-2\Results.csv"
    # path_TEM_25_07_24 = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\TEM\24_06_25\24_06_25\IONP 25_07_24-30c-2\Results.csv"
    # path_TEM_300 = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\TEM\24_06_25\24_06_25\IONP 300\Results.csv"
    # path_TEM_301 = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\TEM\24_06_25\24_06_25\IONP 301\Results.csv"
    # path_TEM_2504 = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\TEM\24_06_25\24_06_25\MP 25_04-2\Results.csv"
    # prd.plot_TEM_analysis(path_TEM_2504, x_scale=[2,15])
