import os
import sys

# Force UTF-8 output for Windows
sys.stdout.reconfigure(encoding='utf-8')

exclude_dirs = {
    'venv', 'SimStack.venv', 'env', '__pycache__', '.git', 
    'node_modules', 'dist', 'build', '.pytest_cache',
    '.mypy_cache', 'eggs', '*.egg-info'
}

def print_tree(root, prefix=""):
    entries = [e for e in os.scandir(root) 
               if e.name not in exclude_dirs 
               and not e.name.endswith('.pyc')
               and not e.name.startswith('.')]
    entries = sorted(entries, key=lambda e: (e.is_file(), e.name))
    
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries)-1 else "├── "
        print(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "    " if i == len(entries)-1 else "│   "
            print_tree(entry.path, prefix + extension)

print_tree(".")