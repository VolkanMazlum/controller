#!/bin/bash
for i in {1..15}
do
    echo "Running iteration $i..."
    mpirun -np 4 music complete.music
    echo "Iteration $i completed."
done
