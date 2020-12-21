#!/bin/sh
# Absolute path to this script. /home/user/bin/foo.sh
SCRIPT=$(readlink -f $0)
# Absolute path this script is in. /home/user/bin
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH

python job1.py
python job2.py
python job3.py
python job4.py
python job5.py
python job6.py
python job7.py
