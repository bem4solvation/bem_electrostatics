import bem_electrostatics.solute

try:
    from simtk import openmm, unit
    from simtk.openmm.app import PDBFile
except ImportError:
    _has_openmm = False
else:
    _has_openmm = True


def create_solute(solute_file_path, external_mesh_file=None, save_mesh_build_files=False,
                  mesh_build_files_dir="mesh_files/", mesh_density=1.0, nanoshaper_grid_scale=None,
                  mesh_probe_radius=1.4, mesh_generator="nanoshaper", print_times=False, force_field="amber"):

    file_extension = solute_file_path.split(".")[-1]
    if file_extension != "pdb":
        raise ValueError('Unrecognised file extension: %s  -> A PDB must be given' % file_extension)

    pqr_file_path = 'temp.pqr'
    apply_forcefield_generate_pqr(solute_file_path, pqr_file_path, forcefield_choice='amber14-all.xml',
                                  remove_extras_from_pdb=True)

    solute_object = bem_electrostatics.Solute(pqr_file_path)

    return solute_object


def simulation_to_solute(simulation):
    simulation.hello()

    solute_object = bem_electrostatics.Solute(generated_pqr_file,
                                              external_mesh_file,
                                              save_mesh_build_files,
                                              mesh_build_files_dir,
                                              mesh_density,
                                              nanoshaper_grid_scale,
                                              mesh_probe_radius,
                                              mesh_generator,
                                              print_times,
                                              force_field
                                              )

    return solute_object


def apply_forcefield_generate_pqr(pdb_file_path, pqr_file_path, forcefield_choice='amber14-all.xml', remove_extras_from_pdb=True):

    pdb = create_pdb_object(pdb_file_path)
    forcefield = openmm.app.ForceField(forcefield_choice)

    pdb_H = openmm.app.modeller.Modeller(pdb.topology, pdb.positions)
    pdb_H.addHydrogens(forcefield, pH=7.0)

    system = forcefield.createSystem(pdb_H.topology)
    force_parameters = system.getForces()[3]

    positions = pdb_H.getPositions()
    topology = pdb_H.getTopology()

    positions_in_angstroms = positions.value_in_unit(unit.angstrom)
    atoms = topology.atoms()

    data = []
    max_lengths = [0] * 10
    for atom in atoms:
        charge = force_parameters.getParticleParameters(atom.index)[0].value_in_unit(unit.elementary_charge)
        radius = force_parameters.getParticleParameters(atom.index)[1].value_in_unit(unit.angstrom)

        position = []
        for i in range(3):
            position.append(str(positions_in_angstroms[atom.index][i]))

        data_ind = ['ATOM', str(atom.id), atom.name, atom.residue.name, str(atom.residue.id), str(position[0]),
                    str(position[1]), str(position[2]), str(charge), str(radius)]
        data.append(data_ind)

        for i in range(len(data_ind)):
            if len(data_ind[i]) > max_lengths[i]:
                max_lengths[i] = len(data_ind[i])

    pqr_lines = []
    for data_ind in data:
        string = ''
        for i in range(len(data_ind)):
            string = string + "{:>{width}}".format(data_ind[i], width=max_lengths[i] + 2)
        pqr_lines.append(string)

    write_file_with_lines(pqr_file_path, pqr_lines)


def create_pdb_object(solute_file_path):
    file_extension = solute_file_path.split(".")[-1]
    if file_extension == "pdb":
        pdb = PDBFile(solute_file_path)
        return pdb
    else:
        raise ValueError('Unrecognised file extension: %s  -> A PDB must be given' % file_extension)


def write_file_with_lines(file_path, lines):
    file = open(file_path, 'w')
    for line in lines:
        file.write(line+"\n")
    file.close()


def remove_extra_atoms_from_pdb(pdb_file_path):
    file = open(file_path, 'w')
    for line in lines:
        file.write(line + "\n")
    file.close()
