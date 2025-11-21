#!/usr/bin/zsh

# This script is used to start all benchmark services in the container
# loop over all dirs in /opt/BencherBenchmarks and execute poetry run start-benchmark-service for each

for dir in ./*; do
    if [ -d "$dir" ]; then
        echo "Starting benchmark service for $dir"
        cd $dir
        #poetry run start-benchmark-service &
        bash -c "uv run start-benchmark-service &"
        cd ..
    fi
done