import copy
import procedure as prd
import numpy as np
import lmfit as lmf


class TGA_dataset:
    def __init__(self):
        self.y_data = None
        self.time = None
        self.temp = None

def calculate_process(scale, G, m, x):
    return scale*(np.exp(G*(m-x)))/(1+np.exp(G*(m-x)))

def calc_TGA(params, x, return_proccs = False):
    all_proc = list()
    num_of_proc = prd.num_of_procc(params)
    bkg_line = len(x)*[params[f'bkg'].value]
    all_proc.append(np.array(bkg_line))
    for i in range(num_of_proc):
        proc = calculate_process(params[f'scale_{i}'],
                          params[f'G_{i}'],
                          params[f'm_{i}'],
                          x
                          )
        all_proc.append(proc)
    if not return_proccs:
        return sum(all_proc)
    else:
        return (all_proc)

def create_params(number, end_weight, x = None):
    #go backwards from end weight creating parametrs

    scale_list = list()
    G_list = list()
    m_list = list()

    for i in range(number):
        scale = 1/number
        G = 0.01
        m = 0
        if x is None:
            m = 300
        else:
            m = 0.9*max(x)/(i+1)
        scale_list.append(scale)
        m_list.append(m)
        G_list.append(G)
    #reversing list to not go backwards
    scale_list = list(reversed(scale_list))
    m_list = list(reversed(m_list))
    G_list = list(reversed(G_list))
    bck = end_weight

    #creating lfm params
    # i = 0 is bkg!

    params = lmf.Parameters()
    params.add(f'bkg', value=bck, min=0.0, max=0.98, vary = False)
    for idx in range(len(scale_list)):
        params.add(f'scale_{idx}', value=scale_list[idx],min=0.0, max = 1)
        params.add(f'm_{idx}', value=m_list[idx],min=0.0)
        params.add(f'G_{idx}', value=G_list[idx],min=0.0)


        if x is not None:
            params[f'm_{idx}'].set(m_list[idx], max=(max(x)), min=(min(x)))

    return params


def chi2_calc(y_data, y_calc, err=None, return_residual=False):
    # chi2 = sum(((y_calc - y_data)/sig)^2)
    chi2 = 0.0
    residual = np.empty(0)
    chi_point_noerr = []
    for i in range(len(y_data)):
        if err is not None:
            weights = copy.deepcopy(err)
            chi_point = ((y_calc[i] - y_data[i]) / weights[i]) ** 2
            chi_point_noerr.append(((y_calc[i] - y_data[i]) / 1) ** 2)
            if return_residual:
                residual = np.append(residual, chi_point)
            else:
                chi2 += chi_point

        else:
            sample = y_data[i], y_calc[i]
            sig = np.sqrt(np.power(y_data[i] - np.mean([y_data[i], y_calc[i]]), 2) / 2)
            diff = (y_calc[i] - y_data[i])
            if return_residual:
                residual = np.append(residual, 100*diff)
            else:
                chi2 += (diff) ** 2

    if return_residual:
        return residual
    else:
        if np.isnan(chi2):
            chi2 = 1e8
        if chi2 is None:
            chi2 = 1e8
        return chi2



def objective(params, TGA_dataset, return_residual = True, ageinst_temp = False):

    x = TGA_dataset.time
    if ageinst_temp:
        x = TGA_dataset.temp

    calculated_TGA = calc_TGA(params, x)
    res = chi2_calc(TGA_dataset.y_data,calculated_TGA, return_residual = return_residual)
    total_res = np.nan_to_num(res, copy=False, nan=1e9)
    print('Chi2: ',sum(total_res)**2)
    return total_res
