class TreeNode:
    def __init__(self, name, parent=None, edge_label=None):
        self.name = name
        self.parent = parent
        self.edge_label = edge_label if parent is not None else None
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def _post_detach(self, parent):
        self.edge_label = None

    def __repr__(self):
        return f"Parent Value: {self.edge_label} --- Name: {self.name} === Children: {self.children}"
