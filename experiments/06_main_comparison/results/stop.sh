#!/bin/bash
# Stops the runner and every training job (matches by PID list, never by our own command line).
for p in $(ps -eo pid,args | grep "[r]un_suite.py" | awk "{print \$1}"); do kill $p; done
sleep 2
for p in $(ps -eo pid,args | grep "[b]in/python train.py\|[p]ython -u train.py" | awk "{print \$1}"); do kill $p; done
sleep 4
echo "still running: $(ps -eo args | grep -c "[t]rain.py")"
