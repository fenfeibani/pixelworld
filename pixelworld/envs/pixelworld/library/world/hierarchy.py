"""
A world containing objects in three-level hierarchies.
"""

objects = [['complex', dict(children=[
                ['complex', dict(parent=0, shape='XXXX', position=(10, 10))],
                ['complex', dict(parent=0, shape='XXXX', position=(13, 10))]])]]

# no parent color specified, so children can have any color
objects += [['complex', dict(children=[
                ['complex', dict(parent=0, color=3, shape='XXXX', position=(15, 10))],
                ['complex', dict(parent=0, color=4, shape='XXXX', position=(18, 10))]])]]

# since it is specified, parent color overrides child color
objects += [['complex', dict(color=5, children=[
                ['complex', dict(parent=0, color=1, shape='XXXX', position=(5, 15))],
                ['complex', dict(parent=0, color=2, shape='XXXX', position=(8, 15))]])]]

objects += [['self_big', dict(color=2, children=[
                ['complex', dict(parent=0, shape='XXXX', position=(5, 10))],
                ['complex', dict(parent=0, shape='XXXX', position=(8, 10))]])]]

