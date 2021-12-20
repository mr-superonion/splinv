# Change the Dir variable to the absolute path of the current directory

Dir="$homeSys/code/massMap_Private/"

export PYTHONPATH="$Dir/ddmap/":$PYTHONPATH

export PATH="$Dir/tasks/sim/":$PATH
export PYTHONPATH="$Dir/tasks/sim/":$PYTHONPATH

export PATH="$Dir/tasks/s19a/":$PATH
export PYTHONPATH="$Dir/tasks/s19a/":$PYTHONPATH

DirBase="$homeSys/code/FPFS_Private/FPFSBASE/"
export PYTHONPATH="$DirBase/python/":$PYTHONPATH
