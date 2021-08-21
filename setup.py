from distutils.core import setup

setup(
    name='bem_electrostatics',
    version='0.1',
    packages=['bem_electrostatics', 'bem_electrostatics.utils', 'bem_electrostatics.openmm',
              'bem_electrostatics.mesh_tools', 'bem_electrostatics.pb_formulation',
              'bem_electrostatics.pb_formulation.formulations'],
    url='',
    license='',
    author='Stefan',
    author_email='stefan.search.14@sansano.usm.cl',
    description='Python library for use with the bempp-cl library, for the solving of the PB implicit solvent model using BEM.',
    include_package_data=True,
)
