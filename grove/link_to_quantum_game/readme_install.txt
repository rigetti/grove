you will need to install blender for the quantum blockout game to work.
download from blender.org

blender is bundled with python 3.5 and the easiest way to access pyquil is to install it into the 
blender directory structure using pip.

pip is not part of Python so it doesn't come by default with Blender.

It has to be installed for Blender's bundled Python even if you already have pip for some 
other version of Python on your system.

For this get the get-pip.py file from the pip documentation

You'll find the blender python binary at:

/blender-path/2.xx/python/bin/python
Use this binary to run the get-pip.py. If you have it saved in your home directory the command to use should look something like this:

$ /path/to/blender/blender-path/2.xx/python/bin/python3 ~/get-pip.py
You should now have pip installed for blender. You use it with blenders python too and you have to point to the pip that was installed for blenders python. Both are in blenders folder tree and you use them like this:

$ /blender-path/2.xx/python/bin/python pip install qutip
$ /blender-path/2.xx/python/bin/python pip install pyquil

This installs qutip and pyquil so that can be accessed within blenders python. 

