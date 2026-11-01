"""Refresh only the named public dataset files from UCI; no arbitrary extraction."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def main():
    destination = Path(__file__).resolve().parents[1] / "data" / "uci"
    destination.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=60) as response:
        archive = response.read()
    with ZipFile(BytesIO(archive)) as outer:
        with ZipFile(BytesIO(outer.read("student.zip"))) as inner:
            for name in ("student-mat.csv", "student-por.csv", "student.txt"):
                (destination / name).write_bytes(inner.read(name))
                print(f"Saved {destination / name}")


if __name__ == "__main__":
    main()
