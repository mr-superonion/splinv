# Install

## Requirements


## Setup the environment:

```shell
source setup.sh
```

makeConfigS16a.py
prepareMassMap_s16a.py
parseMockCatBatchRun.py $s17w --output ./ --queue small --job uniC --time 1000000000 --batch-type=pbs --nodes 2 --procs 20 --clobber-versions --do-exec
