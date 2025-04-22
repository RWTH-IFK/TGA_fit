from lammps import lammps

def try_lammps():
    lmp = lammps()
    lmp.file("in.lj")