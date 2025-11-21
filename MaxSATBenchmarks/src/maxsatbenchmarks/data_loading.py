import os
import pathlib


def download_maxsat60_data(
        dirname: str
):
    if not pathlib.Path(os.path.join(dirname, "frb10-6-4.wcnf")).exists():
        print("frb10-6-4.wcnf not found. Downloading...")

        url = "http://bounce-resources.s3-website-us-east-1.amazonaws.com/wms_crafted.tgz"
        print(f"Downloading {url}")

        import requests
        response = requests.get(url, verify=False)

        with open(os.path.join(dirname, "wms_crafted.tgz"), "wb") as file:
            file.write(response.content)

        import tarfile

        with tarfile.open(os.path.join(dirname, "wms_crafted.tgz"), "r:gz") as tar:
            tar.extractall(dirname)
            # move data/maxsat/wms_crafted/frb/frb10-6-4.wcnf to data/maxsat/frb10-6-4.wcnf
            pathlib.Path(os.path.join(dirname, "wms_crafted/frb/frb10-6-4.wcnf")).rename(
                os.path.join(dirname, "frb10-6-4.wcnf")
            )

            # delete data/maxsat/wms_crafted (even though it is not empty)
            import shutil

            shutil.rmtree(os.path.join(dirname, "wms_crafted"))
        # delete .tgz file
        pathlib.Path(os.path.join(dirname, "wms_crafted.tgz")).unlink()
        print("Data extracted!")


def download_maxsat125_data(
        dirname: str
):
    if not pathlib.Path(
            os.path.join(dirname, "cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf")
    ).exists():
        print(
            "cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf not found. Downloading..."
        )
        import requests

        url = "http://bounce-resources.s3-website-us-east-1.amazonaws.com/mse18-new.zip"
        print(f"Downloading {url}")

        response = requests.get(url, verify=False)

        with open(os.path.join(dirname, "ce.zip"), "wb") as file:
            file.write(response.content)

        import zipfile

        with zipfile.ZipFile(os.path.join(dirname, "ce.zip"), "r") as zip_ref:
            zip_ref.extractall(dirname)

        # extract data/maxsat/mse18-new/cluster-expansion/benchmarks/IS1_5.0.5.0.0.5_softer_periodic.wcnf.gz
        import gzip
        import shutil

        with gzip.open(
                os.path.join(
                    dirname,
                    'mse18-new',
                    'cluster-expansion',
                    'benchmarks',
                    "IS1_5.0.5.0.0.5_softer_periodic.wcnf.gz"
                ),
                "rb",
        ) as f_in:
            # save to data/maxsat/cluster-expansion-IS1_5.wcnf
            with open(
                    os.path.join(dirname, "cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf"),
                    "wb",
            ) as f_out:
                shutil.copyfileobj(f_in, f_out)

        shutil.rmtree(os.path.join(dirname, "mse18-new"))

        # delete .zip file
        pathlib.Path(os.path.join(dirname, "ce.zip")).unlink()
        print("Data extracted!")
