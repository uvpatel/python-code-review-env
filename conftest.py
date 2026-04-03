# Root-level conftest.py – establishes pytest rootdir so that the package's
# __init__.py (which uses relative imports) is not imported as a plain module
# during test collection.
collect_ignore = ["__init__.py"]
