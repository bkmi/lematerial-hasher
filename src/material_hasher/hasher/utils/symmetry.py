from shutil import which

from aflow_xtal_finder import XtalFinder
from monty.tempfile import ScratchDir
from pymatgen.core import Structure


class AFLOWSymmetry:
    """
    AFLOW label adapter class
    """

    def __init__(self, aflow_executable: str = None):
        """AFLOW Symmetry

        Args:
            aflow_executable (str, optional): AFLOW executable. If none listed tries to find aflow in system path

        Raises:
            RuntimeError: If AFLOW is not found
        """

        self.aflow_executable = aflow_executable or which("aflow")

        print("aflow found in {}".format(self.aflow_executable))

        if self.aflow_executable is None:
            raise RuntimeError(
                "Requires aflow to be in the PATH or the absolute path to "
                f"the binary to be specified via {self.aflow_executable=}.\n"
            )

    def aflow_label(self, structure: Structure, tolerance: float = 0.1) -> str:
        """
        Returns AFLOW label for a given structure
        Args:
            structure (Structure): structure to run AFLOW on
            tolerance (float, optional): AFLOW symmetry tolerance. Defaults to 0.1.

        Returns:
            str: AFLOW label
        """

        xtf = XtalFinder(self.aflow_executable)
        with ScratchDir("."):
            structure.to_file("POSCAR")
            data = xtf.get_prototype_label(
                [
                    "POSCAR",
                ],
                options="--tolerance={}".format(tolerance),
            )
        return data
