#!/usr/bin/env python3
# fix_imports.py - Script to fix import paths in the project

import os
import sys
import re

def fix_imports_in_file(file_path):
    """Fix import paths in a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace direct imports from utils with src.utils
    content = re.sub(r'from utils\.', 'from src.utils.', content)
    content = re.sub(r'import utils\.', 'import src.utils.', content)
    
    # Add path setup at the beginning of the file if not already present
    path_setup = """
# Add the project root to the path for relative imports
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
"""
    
    # Check if the file already has path setup
    if 'project_root = os.path.dirname' not in content and 'src_dir = os.path.dirname' not in content:
        # Find the position after imports
        import_section_end = 0
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_section_end = i + 1
        
        # Insert path setup after imports
        if import_section_end > 0:
            lines.insert(import_section_end, path_setup)
            content = '\n'.join(lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Fixed imports in {file_path}")

def find_python_files(directory):
    """Find all Python files in a directory and its subdirectories."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def main():
    """Main function to fix imports in all Python files."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Find all Python files in the project
    python_files = find_python_files(project_root)
    
    # Fix imports in each file
    for file_path in python_files:
        fix_imports_in_file(file_path)
    
    print(f"\n✓ Fixed imports in {len(python_files)} Python files")
    print("Please run the application again to verify the fixes")

if __name__ == "__main__":
    main() 