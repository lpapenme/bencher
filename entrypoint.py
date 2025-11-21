import time

from pathlib import Path

import os
import subprocess
import threading


# please run my action, GitHub

class ServiceThread(threading.Thread):
    def __init__(
            self,
            service_dir: str
    ):
        threading.Thread.__init__(self)
        self.dir = service_dir

    def run(
            self
    ):
        try:
            print(f"Starting service in directory {Path(self.dir).absolute()}")
            # logfile in home directory
            outfile = os.path.join(os.environ["HOME"], "bencher.out")
            errfile = os.path.join(os.environ["HOME"], "bencher.err")
            with open(outfile, 'a+') as out, open(errfile, 'a+') as err:
                subprocess.check_call(
                    ["uv", "run", "start-benchmark-service"],
                    stdout=out,
                    stderr=err,
                    cwd=self.dir,
                    env=os.environ,
                )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Service failed in directory {self.dir}") from e


if __name__ == '__main__':
    bencher_dir = os.path.join("/opt", "bencher")
    # bencher_dir = "."

    threads = []
    for service_dir in os.listdir(bencher_dir):
        # check if dir and pyproject.toml exists
        if os.path.isdir(os.path.join(bencher_dir, service_dir)) and os.path.isfile(
                os.path.join(bencher_dir, service_dir, "pyproject.toml")
        ):
            thread = ServiceThread(os.path.join(bencher_dir, service_dir))
            thread.start()
            threads.append(thread)

    # check threads every 5 seconds
    while True:
        # check for keyboard interrupt
        try:
            for thread in threads:
                if not thread.is_alive():
                    print(f"Thread {thread.dir} is dead. Exiting...")
                    exit(1)
            # sleep for 5 seconds
            time.sleep(5)
        except KeyboardInterrupt:
            print("Keyboard interrupt. Exiting...")
            exit(1)
        # sleep for 5 seconds
        time.sleep(5)
