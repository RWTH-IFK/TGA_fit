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


def do_fit(path, x_cutoff = 161, end_weight = 0.67, n_proc = 6, x_temp = False):

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

    params = ff.create_params(n_proc, end_weight, x_axis)
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
    # do_fit(path, x_cutoff=44, n_proc=4, end_weight=0.895, x_temp = False)
    # do_fit(path, x_cutoff=44, n_proc=4, end_weight=0.8971, x_temp = True)
    # plotting()

    # NP = calc.Nanoparticle(ligand_type='citrate', composition_NP='Fe3O4', wt_NP=ufloat(86.7, 0.5),
    #                        wt_water=ufloat(3.5, 0.27), diameter_NP=ufloat(11.77, 0.7))
    # NP.calculate_sphere()
    # NP.calculate_trunc_octahedron()
    # NP.generate_report(save=False)

    # sim.tryout()
    # MD.try_lammps()
    path_cif = r"C:\Users\admin-lap143\RWTH\Students\Evgeny\IONP\XRD\magnetite.cif"
    # path_cif = r"C:\Users\admin-lap143\Downloads\Telegram Desktop\Fe3O4.cif"
    # a = sim.IONPs_surface(10, binding_indices=[19,18], ligand_offset=1, to_view = ['model'], optimize_rotation = True, miller=(0,0,1), tol = 0.5, path = path_cif)
    # a = sim.IONPs_surface(5, binding_indices=[19], ligand_offset=1, to_view=['model'], optimize_rotation = True, miller=(1,1,1))
    # a = sim.IONPs_surface(2.0, binding_indices=[18, 19, 20], ligand_offset=1.4, to_view=['model'],
    #                       optimize_rotation=True, alligne=False)
    # io.write('result_4.xyz', a, format='xyz')

    #TEM
    path_TEM_cit = r"C:\Users\admin-lap143\RWTH\Students\Evgeny\TEM\dropcasting in water\IONP 13-06\new\Results10.csv"
    path_TEM_deg = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\DEG_samples\TEM\IONP_06.02\TEM Batalov\IONP_06-02-25\Results10-12.csv"
    path_TEM_deg_old = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\DEG_samples\TEM\MP 23-10\MP 23-10\Results1.csv"
    # prd.plot_TEM_analysis(path_TEM_deg)
    prd.plot_TEM_analysis(path_TEM_cit)
