"""User-facing entry point: runs SeDPO only, never runs or requires CoopES."""
import sys
from evaluate_pair import main


if __name__ == '__main__':
    sys.argv.append('--sedpo-only')
    main()
