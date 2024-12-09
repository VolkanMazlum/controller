#!/bin/bash

# Command to execute
COMMAND="mpirun -np 4 music complete.music"

# Loop to run the command 10 times
for i in {1..10}
do
    echo "Running iteration $i..."
    $COMMAND
    echo "Iteration $i completed."
    echo ""
done