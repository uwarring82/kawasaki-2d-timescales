"""kawasaki2d — 2D Kawasaki-Ising coarsening demonstrator.

See the task card ``TASK-kawasaki-mpemba-boundary-v4.md`` for the governing
specification and ``docs/physics.md`` for conventions and derivations.
"""

__version__ = "1.0.1"

import math

# Critical temperature of the 2D square-lattice Ising model, J = k_B = 1.
T_C = 2.0 / math.log(1.0 + 2.0**0.5)  # ≈ 2.269185
