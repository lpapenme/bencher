#!/usr/bin/zsh

# Loop over all dirs in /opt/BencherBenchmarks (or wherever)
for dir in ./*; do
    if [ -d "$dir" ] && [ -f "$dir/pyproject.toml" ]; then
        echo "Starting benchmark service for $dir"

        # Check if venv executable exists
        if [ -f "$dir/.venv/bin/start-benchmark-service" ]; then
            # Run the binary inside the venv directly
            $dir/.venv/bin/start-benchmark-service &
        else
            echo "Error: No venv binary found in $dir/.venv/bin/"
        fi
    fi
done

# Keep the script running to prevent container exit if used as entrypoint
wait