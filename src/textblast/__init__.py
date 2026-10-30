"""textblast: text reuse detection with a patched NCBI BLAST engine."""

import logging

__version__ = "0.9.1"

# A library never configures output handlers; the CLI does (see textblast.logs.setup_logging).
logging.getLogger("textblast").addHandler(logging.NullHandler())


def open_run(path):  # type: ignore[no-untyped-def]
    """Open a run directory or a bundle (see :mod:`textblast.api`)."""
    from .api import open_run as _open_run

    return _open_run(path)
