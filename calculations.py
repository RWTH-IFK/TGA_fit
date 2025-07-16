import numpy as np
from uncertainties import ufloat
import procedure as prd

'''class containes TGA description of NP and have 2 ways to calculate surface density:
for a sphere or for a truncated octahedron NP
'''


class Constants:
    def __init__(self):
        self.Na = 6.02214E+23
        self.M_H2O = 18.01528
        self.single_H2O_mass = self.M_H2O / self.Na


class Nanoparticle():
    '''numbers obtained from TG are passed as ufloat to propogate errors'''

    def __init__(self, composition_NP='Fe3O4', ligand_type='citrate', diameter_NP=7, wt_NP=ufloat(50, 1),
                 wt_water=ufloat(5, 0.1)):
        const = Constants()

        self.composition_NP = composition_NP
        self.ligand_type = ligand_type
        self.wt_NP = wt_NP
        self.wt_water = wt_water
        self.wt_ligand = 100 - self.wt_NP - self.wt_water
        if ligand_type == 'citrate':
            self.lig_mol_mass = 189.123  # g/mol
        elif ligand_type == 'DEG':
            self.lig_mol_mass = 106.12  # g/mol
        elif ligand_type == 'LA':
            self.lig_mol_mass = 90.08  # g/mol
        # more cases for ligands can be added here

        if composition_NP == 'Fe3O4':
            self.cryst_density = 5.1  # g/cm3
            self.molar_mass_NP = 231.533
        if composition_NP == 'ZnO':
            self.cryst_density = 5.6  # g/cm3
            self.molar_mass_NP = 81.38
        # more cases for NP can be added here
        self.diameter_NP = diameter_NP  # nm

        self.mass_single_lig = self.lig_mol_mass / const.Na
        self.mass_NP_formula = self.molar_mass_NP / const.Na
        '''params that will be filed when geometric model is executed'''
        self.surface_of_one_NP = 0
        self.vol_of_one_NP = 0
        self.mass_of_one_NP = 0 #g
        self.mass_H2O_per_NP = 0 #g
        self.mass_lig_per_NP = 0 #g
        self.area_powder = 0 #nm2/g
        self.num_H2O_per_NP = 0
        self.num_lig_per_NP = 0
        self.lig_surface_density = 0
        self.water_surface_density = 0

    def grafting_density_calculation(self):
        const = Constants()
        self.mass_of_one_NP = self.vol_of_one_NP * self.cryst_density * 1e-21
        self.mass_H2O_per_NP = self.mass_of_one_NP * self.wt_water / self.wt_NP
        self.mass_lig_per_NP = self.mass_of_one_NP * self.wt_ligand / self.wt_NP
        self.area_powder = self.surface_of_one_NP / self.mass_of_one_NP
        self.num_H2O_per_NP = self.mass_H2O_per_NP / const.single_H2O_mass
        self.num_lig_per_NP = self.mass_lig_per_NP / self.mass_single_lig
        self.lig_surface_density = self.num_lig_per_NP / self.surface_of_one_NP
        self.water_surface_density = self.num_H2O_per_NP / self.surface_of_one_NP

    def calculate_sphere(self):
        self.model = 'sphere'
        self.surface_of_one_NP = 4*np.pi*(self.diameter_NP/2)**2
        self.vol_of_one_NP = (4/3)*np.pi*(self.diameter_NP/2)**3
        self.grafting_density_calculation()


    def calculate_trunc_octahedron(self):
        self.model = 'truncated octahedron'
        self.L_edge = self.diameter_NP/(np.sqrt(10))
        self.surface_of_one_NP = (6*self.L_edge**2)*(1+2*np.sqrt(3))
        self.vol_of_one_NP = (8*self.L_edge**3)*np.sqrt(2)
        self.grafting_density_calculation()

    def generate_report(self, save = False):
        report = f"""
        Nanoparticle Report:
        --------------------
        Composition: {self.composition_NP}
        Ligand Type: {self.ligand_type}
        Shape: {self.model}

        Diameter: {self.diameter_NP} nm
        Weight NP: {self.wt_NP} wt.%
        Weight Water: {self.wt_water} wt.%
        Weight Ligand: {self.wt_ligand} wt.%

        Crystalline Density: {self.cryst_density} g/cm^3
        Molar Mass NP: {self.molar_mass_NP} g/mol

        Mass of Single Ligand: {self.mass_single_lig:.3e} g
        Mass NP Formula: {self.mass_NP_formula:.3e} g
        Surface of One NP: {self.surface_of_one_NP:.3e} nm^2
        Volume of One NP: {self.vol_of_one_NP:.3e} nm^3
        Mass of One NP: {self.mass_of_one_NP:.3e} g
        Mass H2O per NP: {self.mass_H2O_per_NP:.3e} g
        Mass Ligand per NP: {self.mass_lig_per_NP:.3e} g
        Area Powder: {self.area_powder:.3e} nm^2/g
        Number of H2O per NP: {self.num_H2O_per_NP:.3e}
        Number of Ligands per NP: {self.num_lig_per_NP:.3e}
        
        Ligand Surface Density: {self.lig_surface_density:.3e} per nm^2
        Water Surface Density: {self.water_surface_density:.3e} per nm^2
        """
        print(report)
        if save:
            prd.save_report(report, path = 'result/NP_surface_description.txt')





