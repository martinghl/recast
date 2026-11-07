"""Enable `python -m recast` as an alias for the `recast` console script."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
