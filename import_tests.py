"""
Script for testing correct imports throughout the module.
May be removed and replaced by some automatic test (e.g. using `pytest`).
"""

from pprint import pprint
import SpinWaveToolkit as SWT

pprint(dir(SWT))
pprint(SWT.__all__)
print(f"{SWT.core._class_SingleLayer.MU0 = }")
