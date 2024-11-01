from flytekit import task, ImageSpec
from flytekit.types.file import FlyteFile
import urllib.request
from pathlib import Path
from os import getcwd

# http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv


image = ImageSpec(
    requirements=Path(__file__).parent / "requirements.txt",
)


@task(
    cache_version="1.0",
    cache=True,
    container_image=image,
)
def download_dataset() -> FlyteFile:
    url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
    file_path = "data.tsv"

    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0')
    req.add_header('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8')
    req.add_header('Accept-Language', 'en-US,en;q=0.5')

    r = urllib.request.urlopen(req).read().decode('utf-8')
    with open(file_path, 'w', encoding="utf-8") as f:
        f.write(r)
    return FlyteFile(file_path)
