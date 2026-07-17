import sys
import shutil

print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import pygraphviz
    print("pygraphviz: SUCCESS")
except ImportError as e:
    print("pygraphviz: FAILED -", e)

try:
    import pydot
    print("pydot: SUCCESS")
except ImportError as e:
    print("pydot: FAILED -", e)

# Check sfdp binary
sfdp_path = shutil.which("sfdp")
if sfdp_path:
    print(f"sfdp binary: FOUND at {sfdp_path}")
else:
    print("sfdp binary: NOT FOUND on PATH")
