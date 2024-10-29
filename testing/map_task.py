from flytekit import ImageSpec
from pathlib import Path
from flytekit import task, workflow, map_task

image = ImageSpec(
    requirements=Path(__file__).parent / "requirements.txt",
)


@task(
    container_image=image,
)
def m(in1: str, in2: str) -> str:
    return in1 + in2


@task(
    container_image=image,
)
def make_list(res: list[str]) -> list[str]:
    retVal = []
    for i in res:
        retVal.append(i)
    return retVal


@workflow
def wf() -> list[str]:
    r = map_task(m)(in1=["a", "b", "c"], in2=["1", "2"])
    return r
