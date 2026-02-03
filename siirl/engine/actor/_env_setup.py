# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
"""Environment setup for actor module.

This module sets environment variables that must be configured before importing
torch or other libraries. Import this module first in any file that needs these settings.
"""

import os

# Disable Transformer Engine to avoid ABI compatibility issues
# This is needed when transformer_engine is compiled for a different PyTorch version
os.environ.setdefault("NVTE_FRAMEWORK", "none")
