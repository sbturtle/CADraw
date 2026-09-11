"""CADraw domain package."""

from cadraw import dxf_validation, solver, topology
from cadraw.dxf_validation import DxfValidation, validate_dxf
from cadraw.topology import assemble_open_line_chains

__version__ = "0.1.0"

__all__ = [
    "DxfValidation",
    "__version__",
    "assemble_open_line_chains",
    "dxf_validation",
    "solver",
    "topology",
    "validate_dxf",
]
