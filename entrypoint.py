import time
from pathlib import Path
import os
import subprocess
import threading
import sys
# Use tomllib to parse pyproject.toml (available since Python 3.11)
import tomllib

class ServiceThread(threading.Thread):
    def __init__(self, service_dir: str):
        threading.Thread.__init__(self)
        self.dir = service_dir

    def run(self):
        # 1. Read pyproject.toml to find the script entry point
        abs_dir = Path(self.dir).absolute()
        toml_path = abs_dir / "pyproject.toml"

        try:
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)

            # The script name we are looking for is 'start-benchmark-service'
            script_name = "start-benchmark-service"

            # Extract the module:function definition (e.g., "lassobenchmarks.main:serve")
            script_definition = data["project"]["scripts"][script_name]

            # The definition is 'module.path:function_name'. We only need the module path.
            module_name = script_definition.split(':')[0]

        except FileNotFoundError:
            print(f"Error: pyproject.toml not found in {abs_dir}. Skipping.")
            return
        except KeyError as e:
            print(f"Error: Missing key {e} in {toml_path}. Check [project.scripts]. Skipping.")
            return

        try:
            print(f"Starting service in directory {abs_dir} using module: {module_name}")

            # 2. Define paths to the virtual environment Python interpreter
            venv_bin = abs_dir / ".venv" / "bin"
            python_exe = venv_bin / "python"

            if not python_exe.exists():
                raise FileNotFoundError(f"Python executable not found at {python_exe}. Did uv sync succeed?")

            # The command is: /path/to/.venv/bin/python -m module.path
            cmd = [str(python_exe), "-m", module_name]

            # 3. Prepare environment variables
            env = os.environ.copy()
            env["VIRTUAL_ENV"] = str(abs_dir / ".venv")
            env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")

            # 4. Log files (ensure $HOME is writable, usually /tmp in Apptainer)
            outfile = os.path.join(os.environ.get("HOME", "/tmp"), f"{abs_dir.name}_bencher.out")
            errfile = os.path.join(os.environ.get("HOME", "/tmp"), f"{abs_dir.name}_bencher.err")

            with open(outfile, 'a+') as out, open(errfile, 'a+') as err:
                # 5. Run directly
                subprocess.check_call(
                    cmd,
                    stdout=out,
                    stderr=err,
                    cwd=self.dir,
                    env=env,
                )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Service failed in directory {self.dir}. Command used: {cmd}") from e
        except FileNotFoundError as e:
            # Re-raise FileNotFoundError for the python executable check
            raise Exception(f"Configuration error: {e}")

# The __main__ block remains the same
if __name__ == '__main__':
    bencher_dir = os.path.join("/opt", "bencher")
    # bencher_dir = "."

    threads = []

    if not os.path.exists(bencher_dir):
        print(f"Error: Directory {bencher_dir} does not exist.")
        sys.exit(1)

    for service_dir in os.listdir(bencher_dir):
        full_path = os.path.join(bencher_dir, service_dir)
        # check if dir and pyproject.toml exists
        if os.path.isdir(full_path) and os.path.isfile(os.path.join(full_path, "pyproject.toml")):
            thread = ServiceThread(full_path)
            thread.start()
            threads.append(thread)

    print(f"Started {len(threads)} services.")

    # check threads every 5 seconds
    while True:
        try:
            for thread in threads:
                if not thread.is_alive():
                    print(f"Thread for {thread.dir} is dead. Exiting...")
                    sys.exit(1)
            time.sleep(5)
        except KeyboardInterrupt:
            print("Keyboard interrupt. Exiting...")
            sys.exit(1)