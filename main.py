import numpy as np
import lmfit as lmf
from plotting import plot_tga_ms as plot
import matplotlib.pyplot as plt
import fitting as ff
import procedure as prd
import calculations as calc
from uncertainties import ufloat
def do_fit(path):
    x_temp = False # if false x is time
    return_residual = True
    TGA_dataset = ff.TGA_dataset()
    TGA_dataset.y_data, TGA_dataset.time, TGA_dataset.temp = prd.read_TGA(path, x_cutoff=190)
    end_weight = TGA_dataset.y_data[-10]
    end_weight = 0.8260


    #zoom
    # TGA_dataset.y_data, TGA_dataset.time, TGA_dataset.temp = prd.read_TGA(path, x_cutoff=50)
    # end_weight = 0.9153

    x_axis = TGA_dataset.time
    if x_temp:
        x_axis = TGA_dataset.temp

    params = ff.create_params(7,end_weight,x_axis)
    TGA_dataset.y_data[:] = [x / 100 for x in TGA_dataset.y_data]

    num_iterations = 8000
    result = lmf.minimize(ff.objective, params,
                          args=(
                              #pass to objective
                              TGA_dataset,return_residual, x_temp
                          ),
                          nan_policy='omit', max_nfev=int(num_iterations),  # **fit_kws ampgo
                          )

    #plotting

    report = lmf.fit_report(result.params)
    print(report)
    print(result.message)
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


    path = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\TGA_ligand_count\IONP_20.11_2\data\IONP_20.11-2.txt"
    # do_fit(path)
    # plotting()

    NP = calc.Nanoparticle(ligand_type = 'citrate',wt_NP=ufloat(82.64,0.5),wt_water=ufloat(5.46243,0.27), diameter_NP=ufloat(7,0.5) )
    NP.calculate_sphere()
    # NP.calculate_trunc_octahedron()
    NP.generate_report(save=True)
