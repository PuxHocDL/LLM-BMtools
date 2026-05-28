import ast
import os
import json

class DictAccessVisitor(ast.NodeVisitor):
    def __init__(self):
        self.accesses = []
        self.current_function = None
        self.current_class = None

    def visit_ClassDef(self, node):
        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node):
        prev_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = prev_function

    def visit_Subscript(self, node):
        # We are looking for sequences of string subscripts
        # e.g., api_response["data"]["search_results"]
        # In AST, this is a Subscript whose value is a Subscript, etc.
        
        path = []
        current = node
        while isinstance(current, ast.Subscript):
            slice_val = current.slice
            # Python 3.9+ AST
            if isinstance(slice_val, ast.Constant) and isinstance(slice_val.value, str):
                path.append(slice_val.value)
            elif isinstance(slice_val, ast.Index) and isinstance(slice_val.value, ast.Constant) and isinstance(slice_val.value.value, str):
                path.append(slice_val.value.value)
            elif isinstance(slice_val, ast.Index) and isinstance(slice_val.value, ast.Str):
                # Python < 3.8
                path.append(slice_val.value.s)
            else:
                break
            current = current.value
            
        if path:
            # We collected them from inside out, so reverse
            path.reverse()
            
            # Figure out the base name if possible
            base_name = ""
            if isinstance(current, ast.Name):
                base_name = current.id
                
            if self.current_class and self.current_function == "get_answer":
                self.accesses.append({
                    "class": self.current_class,
                    "base": base_name,
                    "path": path
                })
                
        self.generic_visit(node)


def extract_paths_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    tree = ast.parse(source)
    visitor = DictAccessVisitor()
    visitor.visit(tree)
    
    return visitor.accesses

def run_extraction(tasks_dir, output_file):
    all_paths = {}
    for filename in os.listdir(tasks_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            filepath = os.path.join(tasks_dir, filename)
            try:
                accesses = extract_paths_from_file(filepath)
                for acc in accesses:
                    cls_name = acc["class"]
                    if cls_name not in all_paths:
                        all_paths[cls_name] = []
                    
                    # Convert to dot notation
                    # Note: we don't know the exact array indexing, so we can use [*] for arrays
                    # For simplicity, just join with dots
                    dot_path = ".".join(acc["path"])
                    if dot_path and dot_path not in all_paths[cls_name]:
                        all_paths[cls_name].append(dot_path)
            except Exception as e:
                print(f"Error parsing {filename}: {e}")
                
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_paths, f, indent=2)
        
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tasks_dir = os.path.join(base_dir, "generate_qa_pairs", "tasks")
    output_file = os.path.join(base_dir, "groundtruth_paths.json")
    run_extraction(tasks_dir, output_file)
    print(f"Extracted paths to {output_file}")
