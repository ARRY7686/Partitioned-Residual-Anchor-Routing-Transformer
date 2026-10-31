#!/bin/bash
# usage: kill_tag.sh TAG  -- kills the training process whose RART_TAG is TAG
for p in $(ps -eo pid,args | grep "[b]in/python train.py" | awk "{print \$1}"); do
  if tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "RART_TAG=$1"; then kill $p; echo "killed $p ($1)"; fi
done
