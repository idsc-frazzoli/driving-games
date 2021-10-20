World package
=========================================

This package contains the functions to load the maps in the "maps" folder. Those maps are created the same way as the
ones in the duckietown-world module. First, one defines tiles containing lane segments parametrized with control points
(with orientation) and then one can stack the tiles together (in a yaml file) and load it with the functions available
in this module. In the maps folder, the picture of the map, the lane network as well as the yaml file with the tiles
are found. Furthermore, there are two jupyter notebooks. One that helps to generate new tiles and another on which one
can play around with the functions.

.. automodule:: world
   :members:
   :undoc-members:


Map loading
-----------
Contains the functions that load a map out of the maps folder

.. automodule:: world.map_loading
   :members:


Tiles
-----
Contains the yaml data of all the tiles used to create the maps in the maps folder

.. automodule:: world.tiles
   :members:


Utils
-----
Contains different functions to extract lanes from the maps and to interpolate along them.

.. automodule:: world.utils
   :members:

.. automodule:: world.skeleton_graph
   :members:
