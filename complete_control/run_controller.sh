!#/bin/sh
cd  /home/gambosi/.pyenv/versions/controller/controller/complete_control
mpirun -n 10 music complete.music
cd /home/gambosi/.pyenv/versions/controller/genetic_algorithm
