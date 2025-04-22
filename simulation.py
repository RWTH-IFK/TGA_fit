import copy
import itertools
import os
import sys

import ase
import numpy as np
from ase.calculators.checkpoint import CheckpointCalculator
from ase.cluster.cubic import FaceCenteredCubic
from ase.visualize import view
from ase.optimize import BFGS
from ase import Atoms
from ase.calculators.emt import EMT
from ase import io
import ase.build.supercells as sc
import ase.build as b
import trimesh
from matplotlib import pyplot as plt
from ase.calculators.morse import MorsePotential
from ase.calculators.espresso import Espresso

os.environ['ASE_AIMS_COMMAND'] = 'aims.x'
os.environ['AIMS_SPECIES_DIR'] = '/home/myname/FHIaims/species_defaults/light'
from scipy.spatial import Delaunay, ConvexHull
from ase.build import add_adsorbate, surface, sort
from ase.vibrations import Vibrations
from ase.vibrations import Infrared

from lmfit import Minimizer, Parameters


def plot_shape(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Unzip the points into their respective coordinates
    x_coords, y_coords, z_coords = zip(*points)

    # Plot the points
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()


def trunc_octahedron_simple():
    # Generate all vertices of a truncated octahedron
    coords = set()

    # Vertices of type (0, ±1, ±2) permutations
    for perm in set(itertools.permutations([0, 1, 2])):
        for signs in itertools.product([-1, 1], repeat=3):
            vertex = tuple(perm[i] * signs[i] for i in range(3))
            coords.add(vertex)

    vertices = np.array(list(coords), dtype=float)
    return coords, vertices


def truncated_octahedron(t=0.99, scale=1.0, shift_to_positive=False, shift_offset=(0, 0, 0), stretch=(0, 0, 0)):
    """
    Create a truncated octahedron mesh with adjustable truncation.

    Parameters:
    - t: truncation factor.
        t=0 -> octahedron (6 vertices)
        t=1 -> truncated octahedron (24 vertices, standard)
    - scale: uniform scaling factor

    Returns:
    - mesh: trimesh mesh object
    """

    vertices = []

    # Octahedron vertices (for t=0)
    oct_vertices = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])

    # Truncated vertices (for t=1)
    trunc_vertices = set()
    perms = set(itertools.permutations([0, 1, 2]))
    signs = [-1, 1]

    for perm in perms:
        for sign1 in signs:
            for sign2 in signs:
                v = [0, 0, 0]
                v[perm[0]] = 0
                v[perm[1]] = sign1
                v[perm[2]] = 2 * sign2
                trunc_vertices.add(tuple(v))

    trunc_vertices = np.array(list(trunc_vertices))

    # Interpolate from octahedron toward truncated vertices
    if t == 0:
        vertices = oct_vertices
    elif t == 1:
        vertices = trunc_vertices
    else:
        # At intermediate truncation, we include both octahedron and truncated octahedron vertices,
        # interpolated towards the truncated vertices
        vertices = []
        for ov in oct_vertices:
            for tv in trunc_vertices:
                # they share the same principal axis alignment if dot(ov, tv) != 0
                if np.dot(ov, tv) > 0 and np.count_nonzero(tv) == 2:
                    interpolated_vertex = (1 - t) * ov + t * tv
                    vertices.append(interpolated_vertex)
        vertices = np.unique(np.round(vertices, decimals=8), axis=0)

    vertices *= scale
    if shift_to_positive:
        min_x, min_y, min_z = np.min(vertices, axis=0)
        print((min_x, min_y, min_z))
        for point in vertices:
            point[0] += abs(min_x) + shift_offset[0]
            point[1] += abs(min_y) + shift_offset[1]
            point[2] += abs(min_z) + shift_offset[2]
            print(point)

    return vertices


def is_point_inside_truncated_octahedron(point, dimensions):
    """
    Check if a point [x,y,z] is inside a truncated octahedron
    defined by lengths (a, b, c) along x, y, z.

    Parameters:
        point: np.array or list, the coordinates [x,y,z]
        dimensions: tuple (a, b, c), dimensions along x, y, z axes.

    Returns:
        True if point is inside truncated octahedron, False otherwise.
    """
    a, b, c = dimensions
    x, y, z = point
    xa, yb, zc = abs(x / a), abs(y / b), abs(z / c)

    # These two inequalities define the standard truncated octahedron shape clearly
    if (xa + yb + zc <= 1.5) and (xa <= 1 and yb <= 1 and zc <= 1):
        return True
    else:
        return False


def cut_out_shape(cell, shape):
    cut_cell = copy.deepcopy(cell)
    cut_indexes = list()
    for i, atom in enumerate(cut_cell.positions):

        isin = Delaunay(shape).find_simplex(atom) >= 0
        if not isin:
            cut_indexes.append(i)

    del cut_cell[cut_indexes]

    return cut_cell


def align_vectors(v1, v2):
    """
    Compute a rotation matrix that aligns vector v1 to vector v2.
    Parameters:
      v1, v2: 3-element arrays or lists.
    Returns:
      R: 3x3 rotation matrix such that R @ (normalized v1) approximates normalized v2.
      rmsd: For our purposes, always 0 if the rotation is done exactly.
    """
    a = np.array(v1, dtype=float) / np.linalg.norm(v1)
    b = np.array(v2, dtype=float) / np.linalg.norm(v2)
    c = np.dot(a, b)

    if np.isclose(c, 1.0):
        return np.eye(3), 0.0
    if np.isclose(c, -1.0):
        if (abs(a[0]) < abs(a[1])) and (abs(a[0]) < abs(a[2])):
            tmp = np.array([1, 0, 0], dtype=float)
        elif (abs(a[1]) < abs(a[2])):
            tmp = np.array([0, 1, 0], dtype=float)
        else:
            tmp = np.array([0, 0, 1], dtype=float)
        u = np.cross(a, tmp)
        u = u / np.linalg.norm(u)
        R = -np.eye(3) + 2 * np.outer(u, u)
        return R, 0.0

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    a_rot = R.dot(a)
    rmsd = np.sqrt(np.mean((a_rot - b) ** 2))
    return R, rmsd


def select_equidistant_points(candidates, candidate_face_indices, n_points):
    """
    Greedily select n_points from candidates such that they are far apart.
    Parameters:
    candidates: (N,3) array of candidate points.
    candidate_face_indices: array-like, the face index for each candidate.
    n_points: desired number of selected points.
    Returns:
    sel_points: (n_points,3) array of selected points.
    sel_face_indices: list (or array) of corresponding face indices.
    """
    n_candidates = len(candidates)
    if n_candidates == 0 or n_points < 1:
        return np.array([]), []
    # Start with the first candidate:
    selected_indices = [0]
    # Initialize distances (set to infinity for all candidates)
    dists = np.full(n_candidates, np.inf)
    # Greedily choose the candidate that maximizes the minimum distance.
    for _ in range(1, n_points):
        last_index = selected_indices[-1]
        last_point = candidates[last_index]
        # Update distances from the last selected point:
        diff = candidates - last_point
        new_d = np.linalg.norm(diff, axis=1)
        dists = np.minimum(dists, new_d)
        # Mark already selected points so they are ignored:
        for idx in selected_indices:
            dists[idx] = -1
        next_idx = int(np.argmax(dists))
        # If all remaining distances are negative, break out
        if dists[next_idx] < 0:
            break
        selected_indices.append(next_idx)

    sel_points = candidates[selected_indices]
    # In case candidate_face_indices is not an array, use a list comprehension.
    try:
        sel_face_indices = candidate_face_indices[selected_indices]
    except:
        sel_face_indices = [candidate_face_indices[i] for i in selected_indices]
    return sel_points, sel_face_indices


def graft_molecules(nanoparticle, adsorbate, grafting_density, offset=2.0, orientation='any', adsorb_to='any', tol=1.0):
    '''
    Graft adsorbate molecules uniformly on the nanoparticle surface.
    Equidistant placement is enforced. The adsorption site is chosen either
    by sampling the convex-hull (adsorb_to='any') or by selecting surface atoms
    of the specified element (e.g. adsorb_to='Fe' or 'O'). In addition, the orientation
    option allows you to choose the bonding direction: use 'any' to align the molecule’s
    +z axis with the local normal, or 'perpendicular' to align the molecule’s principal
    (elongated) axis with the normal.
    tol in Å; how far from surface atom considered as surface atom
    Parameters:
      nanoparticle: ASE Atoms object representing the nanoparticle.
      adsorbate: ASE Atoms object for the molecule to graft.
                Its bonding direction is assumed to be along +z if orientation is 'any',
                or along its principal long axis if orientation is 'perpendicular'.
      grafting_density: Desired density in [molecules/nm²].
      offset: Distance (in Å) above the surface where the adsorbate’s center-of-mass is placed.
      orientation: Either 'any' (default) or 'perpendicular'.
      adsorb_to: The element on the nanoparticle to which the molecule should bind.
                 If 'any', adsorption sites are chosen arbitrarily from the convex-hull;
                 otherwise, only atoms with the given atomic symbol (e.g. 'Fe' or 'O') will be used.

    Returns:
      new_structure: ASE Atoms object containing the nanoparticle with grafted molecules.
    '''
    positions = nanoparticle.get_positions()
    hull = ConvexHull(positions)
    hull_mesh = trimesh.Trimesh(vertices=positions, faces=hull.simplices, process=False)
    area_nm2 = hull_mesh.area / 100.0  # 1 nm² = 100 Å²
    n_ads = int(round(grafting_density * area_nm2))
    print("Nanoparticle surface area: {:.2f} nm².".format(area_nm2))
    print("Grafting density: {:.2f} molecules/nm² → placing {} molecules.".format(grafting_density, n_ads))
    if n_ads < 1:
        print("The chosen grafting density is too low to add any molecules.")
        return nanoparticle

    # Determine the default bonding direction.
    if orientation.lower() == 'perpendicular':
        com_ads = adsorbate.get_center_of_mass()
        rel_pos = adsorbate.positions - com_ads
        cov_matrix = np.dot(rel_pos.T, rel_pos)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        principal_axis = eigvecs[:, np.argmax(eigvals)]
        default_axis = principal_axis / np.linalg.norm(principal_axis)
    else:
        default_axis = np.array([0, 0, 1])

    new_structure = nanoparticle.copy()
    ads_only = Atoms()
    # -------------------------------------------------------------------------
    # Candidate selection based on a specific element.
    if adsorb_to.lower() != 'any':
        # Instead of taking convex hull vertices (which are extreme corners),
        # determine as candidate all atoms of the desired element that lie close to any hull facet.
        # tol in Å; adjust this tolerance as needed.
        candidate_atom_indices = []
        for i, atom in enumerate(nanoparticle):
            if atom.symbol == adsorb_to:
                p = atom.position
                # hull.equations are in the form (a,b,c,d) for the plane: a*x+b*y+c*z+d=0.
                distances = [abs(np.dot(eq[:3], p) + eq[3]) for eq in hull.equations]
                if min(distances) < tol:
                    candidate_atom_indices.append(i)
        print(f'{len(candidate_atom_indices)} atoms are available on the surface')
        if len(candidate_atom_indices) == 0:
            print("No surface atoms of type {} found within tolerance. Falling back to adsorb_to='any'.".format(
                adsorb_to))
            adsorb_to = 'any'
        else:
            candidate_points = np.array([nanoparticle[i].position for i in candidate_atom_indices])
            sel_n = min(n_ads, len(candidate_points))
            sel_points, sel_tags = select_equidistant_points(candidate_points, candidate_atom_indices, sel_n)
            center = np.mean(positions, axis=0)
            for p in sel_points:
                # Get all hull facets for which the distance is within the tolerance.
                normals = []
                for eq in hull.equations:
                    d = abs(np.dot(eq[:3], p) + eq[3])
                    if d < tol:
                        normals.append(eq[:3])
                if len(normals) == 0:
                    local_normal = p - center
                    local_normal /= np.linalg.norm(local_normal)
                else:
                    local_normal = np.mean(normals, axis=0)
                    local_normal /= np.linalg.norm(local_normal)
                target_anchor = p + offset * local_normal
                ads = adsorbate.copy()
                com = ads.get_center_of_mass()
                rot_matrix, rmsd = align_vectors(default_axis, local_normal)
                new_positions = np.dot(ads.positions - com, rot_matrix.T) + target_anchor
                ads.positions = new_positions
                new_structure += ads
                ads_only += ads
            return new_structure, ads_only

    # -------------------------------------------------------------------------
    # If adsorb_to is 'any', sample arbitrary surface points.
    n_candidates = max(200, 10 * n_ads)
    candidate_points, candidate_face_indices = trimesh.sample.sample_surface(hull_mesh, n_candidates)
    sel_points, sel_face_indices = select_equidistant_points(candidate_points, candidate_face_indices, n_ads)
    for p, face_idx in zip(sel_points, sel_face_indices):
        local_normal = hull_mesh.face_normals[face_idx]
        target_anchor = p + offset * local_normal
        ads = adsorbate.copy()
        com = ads.get_center_of_mass()
        rot_matrix, rmsd = align_vectors(default_axis, local_normal)
        new_positions = np.dot(ads.positions - com, rot_matrix.T) + target_anchor
        ads.positions = new_positions
        new_structure += ads
        ads_only += ads
    return new_structure, ads_only


def delete_surface_atoms(nanoparticle, element, tol):
    '''
    Remove atoms of a specified type that lie within a given distance from the
    nanoparticle surface. The surface is approximated by the convex hull of all
    atomic positions. For each atom of the specified element, the minimum absolute
    distance to any convex hull facet is computed. If this distance is less than or
    equal to the given tolerance, the atom is considered a surface atom and is removed.
    Parameters:
      nanoparticle : ASE Atoms object
          The input nanoparticle.
      element : str
          The atomic symbol (e.g., 'Fe', 'O') of atoms to be removed.
      tol : float
          Tolerance distance (in Å) from the convex hull surface. Atoms with a minimal
          distance <= tol are considered to lie on the surface.

    Returns:
      new_nanoparticle : ASE Atoms object
          A copy of the input nanoparticle with the specified surface atoms removed.
    '''
    import numpy as np
    from scipy.spatial import ConvexHull

    # Create a copy of the nanoparticle so that the original is not modified.
    new_nanoparticle = nanoparticle.copy()
    positions = new_nanoparticle.get_positions()

    # Compute the convex hull; its facets approximate the nanoparticle surface.
    hull = ConvexHull(positions)

    indices_to_remove = []
    # Loop over all atoms and check if an atom of the specified type is on the surface.
    for i, atom in enumerate(new_nanoparticle):
        if atom.symbol == element:
            pos = atom.position
            # For each plane (facet) of the hull, compute its distance to the current atom.
            distances = [abs(np.dot(eq[:3], pos) + eq[3]) for eq in hull.equations]
            # If the smallest distance is within the tolerance, mark the atom for deletion.
            if distances and min(distances) <= tol:
                indices_to_remove.append(i)

    # Remove atoms in reverse order to prevent reindexing issues.
    for i in sorted(indices_to_remove, reverse=True):
        del new_nanoparticle[i]

    return new_nanoparticle


def find_close_atom_indices(atoms, tol):
    '''
    Find pairs of atom indices in an ASE Atoms object that are positioned
    at a distance equal to or less than the given tolerance.
    Parameters:
      atoms : ASE Atoms object.
      tol   : float
              Tolerance distance in Ångströms.

    Returns:
      close_pairs : list of tuples
                    Each tuple (i, j) indicates that atoms with indices i and j
                    are closer than or equal to tol apart.
    '''
    positions = atoms.get_positions()
    n_atoms = len(positions)
    close_pairs = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if np.linalg.norm(positions[i] - positions[j]) <= tol:
                close_pairs.append((i, j))
    return close_pairs


def Whole_particle():
    # supercell
    cif = io.read(r"C:\Users\admin-lap143\RWTH\Students\Evgeny\IONP\XRD\magnetite.cif")
    P = 7.0 * np.diag(np.array([1, 1, 1]))
    super_cell = sc.make_supercell(cif, P)
    # super_cell.calc = MorsePotential()
    # super_cell.get_potential_energy()
    # dyn = BFGS(super_cell,trajectory='IONP_try.traj')
    # dyn.run(fmax=0.05)
    # view(super_cell)

    # shape
    a = truncated_octahedron(scale=15, shift_to_positive=True, shift_offset=(0.0, 0.0, 0.0))
    cut_cell = cut_out_shape(super_cell, a)

    less_O = delete_surface_atoms(cut_cell, 'O', 1)

    # citric acid
    path_CA = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\citric_acid.xyz"
    CA = io.read(path_CA, format='xyz', index=":")[0]
    CA.calc = MorsePotential()

    # adding water
    # d = 0.9575
    # t = np.pi / 180 * 104.51
    # water = Atoms('H2O',
    #               positions=[(d, 0, 0),
    #                          (d * np.cos(t), d * np.sin(t), 0),
    #                          (0, 0, 0)],
    #               calculator=EMT())
    # dyn = BFGS(water)
    # dyn.run(fmax=0.05)

    grafted_NP, ads = graft_molecules(less_O, CA, 5.0, orientation='perpendicular', adsorb_to='Fe', tol=1.3)
    # overlap = find_close_atom_indices(ads, 0.8)
    # print(overlap)
    # print(ads[overlap[0][0]])
    # print(ads[overlap[0][1]])
    # print(ads[overlap[1][0]])
    # print(ads[overlap[1][1]])
    grafted_NP.calc = MorsePotential()
    ads.calc = MorsePotential()
    view(grafted_NP)

    # Vibrations
    # vib = Vibrations(CA)
    # vib.run()
    # vib.summary(log='CA_vibration_summary.txt')
    # vib.write_mode(-1)


def farthest_point_sampling(candidates, num_points):
    """
    Given a list of candidate points (an (N,2) array representing x-y coordinates)
    and the desired number of points, return indices (relative to candidates)
    selected using a farthest-point sampling algorithm.
    """
    if num_points >= len(candidates):
        return list(range(len(candidates)))
    selected = [np.random.choice(len(candidates))]
    candidate_indices = set(range(len(candidates))) - set(selected)

    while len(selected) < num_points:
        best_candidate = None
        best_distance = -1.0
        for i in candidate_indices:
            d = min(np.linalg.norm(candidates[i] - candidates[j]) for j in selected)
            if d > best_distance:
                best_distance = d
                best_candidate = i
        selected.append(best_candidate)
        candidate_indices.remove(best_candidate)
    return selected


def print_binding_atoms(mol, indx):
    for i in indx:
        print(i, mol[i].position)


def align_molecule_with_z(passed_molecule: Atoms, selected_indices: list, override=True):
    """
    Rotates the molecule so that the selected atom(s) are at the lowest Z value,
    and the farthest atom is at the highest Z value.

    :param molecule: ASE Atoms object representing the molecule
    :param selected_indices: List of atom indices that should be at the lowest Z value
    """
    for idx in selected_indices:
        if "bind" not in passed_molecule.arrays:
            passed_molecule.new_array("bind", np.zeros(len(passed_molecule), dtype=int))
        passed_molecule.arrays['bind'][idx] = 1
    if override:
        molecule = passed_molecule
    else:
        molecule = copy.deepcopy(passed_molecule)
    positions = molecule.get_positions()

    # Compute center of selected atoms
    selected_center = np.mean(positions[selected_indices], axis=0)

    # Compute displacement vectors from the selected center
    displacement_vectors = positions - selected_center

    # Find the farthest atom from the selected center
    farthest_index = np.argmax(np.linalg.norm(displacement_vectors, axis=1))
    farthest_vector = displacement_vectors[farthest_index]

    # Define the target direction (Z-axis)
    target_vector = np.array([0, 0, 1])

    # Compute rotation axis using cross product
    rotation_axis = np.cross(farthest_vector, target_vector)
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Compute rotation angle using dot product
    cos_theta = np.dot(farthest_vector, target_vector) / np.linalg.norm(farthest_vector)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # Construct rotation matrix (Rodrigues' rotation formula)
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    # Apply rotation to all positions
    new_positions = np.dot(positions - selected_center, R.T) + selected_center
    molecule.set_positions(new_positions)

    return molecule


def rotate_ligand_z(ligand, theta):
    """
    Rotates a ligand around its own center of mass along the Z-axis by a given angle theta.

    Parameters:
        ligand (ase.Atoms): The ligand (ASE Atoms object) to be rotated.
        theta (float): Rotation angle in radians.

    Returns:
        rotated_ligand (ase.Atoms): The rotated ligand.
    """
    # Compute the center of mass (COM)
    com = ligand.get_center_of_mass()

    # Define the 3D rotation matrix for rotation around the Z-axis
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    # Create a copy of the ligand to avoid modifying the original object
    rotated_ligand = ligand.copy()

    # Rotate each atom's position around the COM
    for atom in rotated_ligand:
        pos = atom.position - com  # Translate to COM
        pos = np.dot(rotation_matrix, pos)  # Rotate
        atom.position = pos + com  # Translate back

    return rotated_ligand



def minimize_overlap(params, ligand, ligand_atom_indices, ads_system):
    """
    Objective function for lmfit, which attempts to find a rotation angle theta
    that maximizes the minimum distance between ligand and existing atoms.

    Parameters:
        params (lmfit.Parameters): Parameters object containing theta (rotation angle).
        ligand (ase.Atoms): The ligand being rotated.
        ligand_atom_indices (list): Indices of previously placed ligand atoms.
        ads_system (ase.Atoms): The full system containing adsorbed ligands.

    Returns:
        float: Negative minimum distance (because lmfit minimizes by default, but we want to maximize distance).
    """
    theta = params['theta'].value
    rotated_ligand = rotate_ligand_z(ligand, theta)

    # Compute minimum distance between rotated ligand and existing ligand atoms
    min_distance = 999999
    for atom in rotated_ligand:
        for idx in ligand_atom_indices:
            distance = np.linalg.norm(atom.position - ads_system[idx].position)
            min_distance = min(min_distance, distance)

    return 1/min_distance  # Inverse because lmfit minimizes, but we want to maximize distance


def check_and_optimize_ligand_overlap(ligand, ligand_atom_indices, ads_system, fe_idx):
    """
    Checks for overlap between the given ligand and previously placed ligands.
    If an overlap is found, it optimizes the ligand's rotation around the Z-axis
    using lmfit to maximize the minimum distance from existing ligand atoms.

    Parameters:
        ligand (ase.Atoms): Ligand to check overlap for.
        ligand_atom_indices (list): Indices of previously placed ligand atoms.
        ads_system (ase.Atoms): The full system containing adsorbed ligands.
        fe_idx (int): Index of the binding Fe atom.

    Returns:
        bool: True if ligand placement is successful, False if overlapping could not be resolved.
    """
    # Check for initial overlaps
    overlap_found = False
    for atom in ligand:
        new_pos = atom.position
        for idx in ligand_atom_indices:
            if np.linalg.norm(new_pos - ads_system[idx].position) < 0.5:
                print(f"Overlap detected: Ligand on Fe index {fe_idx} is too close to an existing ligand atom.")
                overlap_found = True
                break
        if overlap_found:
            break

    # Try to optimize rotation if overlap is found
    if overlap_found:
        print(f"Trying to rotate ligand on Fe index {fe_idx} to avoid overlap.")

        # Set up optimization parameters
        params = Parameters()
        # params.add('theta', value=0.0, min=-np.pi, max=np.pi)  # Rotation range: -π to π
        params.add('theta', value=0.0, min=-180, max=180)
        # Run optimization
        minner = Minimizer(minimize_overlap, params, fcn_args=(ligand, ligand_atom_indices, ads_system))
        result = minner.minimize()

        # Apply the optimal rotation to the ligand
        optimal_theta = result.params['theta'].value
        optimized_ligand = rotate_ligand_z(ligand, optimal_theta)

        # Check if overlap still exists after optimization
        overlap_still_present = False
        for atom in optimized_ligand:
            new_pos = atom.position
            for idx in ligand_atom_indices:
                if np.linalg.norm(new_pos - ads_system[idx].position) < 0.5:
                    overlap_still_present = True
                    break
            if overlap_still_present:
                break

        if overlap_still_present:
            print(f"Skipping ligand on Fe index {fe_idx} due to overlap issues after optimization.")
            return False  # Skipping current ligand placement
        else:
            print(f"Successfully optimized ligand on Fe index {fe_idx} to avoid overlap.")
            ligand.positions[:] = optimized_ligand.positions  # Update ligand positions
            return True  # Ligand placement is successful

    return True  # No overlap detected initially



def IONPs_surface(grafting_density, binding_indices=[0], ligand_offset=2.0, to_view=['all'], rotate=False,
                  optimize_rotation=False, alligne = True, miller =(0, 0, 1),tol = 0.5, path = None) :
    """
    Generates adsorbed ligands on IONP surface.

    Parameters:
        grafting_density (float): Desired ligands per nm².
        binding_indices (list): Indices of ligand atoms that bind to surface.
        ligand_offset (float): Distance (Å) of the binding point above Fe atom.
        to_view (list): List specifying which views to display ('all', 'model', 'ligand').
        rotate (bool): If True, ligands are randomly rotated upon placement.
        optimize_rotation (bool): If True, ligands are rotated to maximize distance before placement.

    Returns:
        ASE Atoms object containing surface with grafted ligands.
    """
    # Load the bulk magnetite surface
    try:
        bulk = io.read(path)
    except Exception as e:
        sys.exit("Error reading magnetite.cif\n" + str(e))


    layers = 1
    surf = surface(bulk, miller, layers, periodic = False)

    surf.center(vacuum=10.0, axis=2)
    surf = sort(surf)

    surf = surf.repeat((3, 3, 1))  # Expand the surface

    # Compute the surface area in nm²
    cell = surf.get_cell()
    area_A2 = np.linalg.norm(np.cross(cell[0][:2], cell[1][:2]))
    area_nm2 = area_A2 / 100.0  # Convert Å² to nm²
    desired_ligands = int(grafting_density * area_nm2)

    # Identify topmost Fe atoms
    z_max = max(atom.position[2] for atom in surf)

    candidate_indices, candidate_xy = [], []
    for i, atom in enumerate(surf):
        if atom.symbol == "Fe" and (z_max - atom.position[2]) < tol:
            candidate_indices.append(i)
            candidate_xy.append(atom.position[:2])

    print(f"Found {len(candidate_indices)} Fe atoms for ligand attachment.")

    # Select Fe binding positions using farthest-point sampling.
    num_sites = min(desired_ligands, len(candidate_indices))
    selected_idx_relative = farthest_point_sampling(np.array(candidate_xy), num_sites)
    selected_Fe_indices = [candidate_indices[i] for i in selected_idx_relative]
    print(f"Selected {len(selected_Fe_indices)} Fe binding sites.")

    # Load the citric acid ligand
    path_CA = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\citric_acid.xyz"
    try:
        citrate_mol = io.read(path_CA, format='xyz')
    except Exception as e:
        sys.exit(f"Error reading citric_acid.xyz\n{str(e)}")

    ads_system = surf.copy()
    ligand_atom_indices = []
    added_ligands = 0
    current_total = len(ads_system)

    # Iterate over selected Fe sites
    for fe_idx in selected_Fe_indices:
        fe_atom = surf[fe_idx]
        ligand = citrate_mol.copy()

        # Determine the binding point of the ligand
        positions = ligand.get_positions()
        try:
            binding_point = np.mean([positions[i] for i in binding_indices], axis=0)
        except IndexError:
            print(f"Error: Invalid binding indices {binding_indices} for ligand.")
            continue

        # Determine the long axis
        distances = np.linalg.norm(positions - binding_point, axis=1)
        farthest_atom_idx = int(np.argmax(distances))
        long_axis_vec = positions[farthest_atom_idx] - binding_point
        if long_axis_vec[2] < 0:
            long_axis_vec = -long_axis_vec

        if np.linalg.norm(long_axis_vec) < 1e-6:
            print(f"Warning: Long-axis vector near zero for Fe index {fe_idx}.")
            continue

        # Align ligand and translate to correct position
        if alligne:
            align_molecule_with_z(ligand, binding_indices)
        target = np.array([fe_atom.position[0], fe_atom.position[1], fe_atom.position[2] + ligand_offset])
        ligand.translate(target - binding_point)

        # Apply Random Rotation (if `rotate=True`)**
        if rotate:
            rand_theta = np.random.randint(0, 180) * (np.pi / 180)  # Convert to radians
            ligand = rotate_ligand_z(ligand, rand_theta)

        # Optimize Rotation with `minimize_overlap` (if `optimize_rotation=True`)**
        if optimize_rotation:
            params = Parameters()
            params.add('theta', value=0.0, min=-np.pi, max=np.pi)  # Rotate between -π and π
            minner = Minimizer(minimize_overlap, params, fcn_args=(ligand, ligand_atom_indices, ads_system))

            result = minner.minimize()


            # Apply the optimized rotation
            optimal_theta = result.params['theta'].value
            ligand_copy = copy.deepcopy(ligand)
            ligand = rotate_ligand_z(ligand, optimal_theta)


            # view(ligand)
            # view(ligand_copy)

        # Check for overlaps; skip ligand if it still overlaps
        new_indices = list(range(current_total, current_total + len(ligand)))

        if not check_and_optimize_ligand_overlap(ligand, ligand_atom_indices, ads_system, fe_idx):
            continue

        # Add ligand to the system
        ads_system += ligand
        added_ligands += 1
        ligand_atom_indices.extend(new_indices)
        current_total = len(ads_system)

    print(f"Grafting completed. Total system atoms: {len(ads_system)}. Ligands added: {added_ligands}. Grafting density: {round(added_ligands/area_nm2, 2)} lig/nm2 ")

    # Viewing options
    if 'all' in to_view:
        print_binding_atoms(citrate_mol, binding_indices)
        view(ads_system)
        view(citrate_mol)
        return ads_system
    elif 'model' in to_view:
        view(ads_system)
    elif 'ligand' in to_view:
        view(citrate_mol)

    return ads_system


def rotation_tryout():
    # Load citric acid ligand.
    path_CA = r"C:\Users\admin-lap143\RWTH\QENS\Organized\synthesis\IONPs_surface_study\Paper\citric_acid.xyz"
    try:
        # Read the first molecule in the XYZ file.
        citrate_mol = io.read(path_CA, format='xyz', index=":")[0]
    except Exception as e:
        sys.exit("Error reading citric_acid.xyz\n" + str(e))

    citrate_mol = align_molecule_with_z(citrate_mol, [-1])
    rotated = rotate_ligand_z(citrate_mol, 30)
    view(rotated)
    view(citrate_mol)
