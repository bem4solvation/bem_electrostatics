from simtk import openmm, unit
from simtk.openmm.app import PDBFile
import bem_electrostatics.solute


def create_solute(solute_file_path, external_mesh_file=None, save_mesh_build_files=False,
                  mesh_build_files_dir="mesh_files/", mesh_density=1.0, nanoshaper_grid_scale=None,
                  mesh_probe_radius=1.4, mesh_generator="nanoshaper", print_times=False, force_field="amber"):

    file_extension = solute_file_path.split(".")[-1]
    if file_extension == "pdb":
        pdb = PDBFile(solute_file_path)
    else:
        raise ValueError('Unrecognised file extension: %s  -> A PDB must be given' % file_extension)

    forcefield = openmm.app.ForceField('amber14-all.xml')
    pdb_H = openmm.app.modeller.Modeller(pdb.topology, pdb.positions)
    pdb_H.addHydrogens(forcefield, pH=7.0)

    system = forcefield.createSystem(pdb_H.topology)

    force_parameters = system.getForces()[3]
    for i in range(system.getNumParticles()):
        print(i)
        print(force_parameters.getParticleParameters(i))

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
