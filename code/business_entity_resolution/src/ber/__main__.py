"""``python -m ber <stage> [--config ...] [--set k=v] [--run ID]`` (see ``ber.cli``)."""

import sys

from .cli import main

sys.exit(main())
