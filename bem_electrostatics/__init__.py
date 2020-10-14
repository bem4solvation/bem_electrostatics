import bempp.api
import os
from bem_electrostatics.solute import Solute
from bem_electrostatics import mesh_tools
from bem_electrostatics import utils
from bem_electrostatics import pb_formulation

BEM_ELECTROSTATICS_PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
# WORKING_PATH = os.getcwd()
