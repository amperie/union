import pandas as pd
from flytekit import task, workflow, ImageSpec
from flytekit import StructuredDataset, Deck
from pathlib import Path
from dataclasses import dataclass

image = ImageSpec(
    requirements=Path(__file__).parent / "requirements.txt",
)


@dataclass
class Compound:
    a: int
    b: float
    df: StructuredDataset


@task(
    container_image=image,
)
def make_data() -> Compound:
    df = pd.DataFrame({"a": [1, 2, 3]})
    return Compound(
        1, 2.0,
        StructuredDataset(dataframe=df)
    )


@task(
    container_image=image,
    enable_deck=True,
)
def pass_data(input: Compound) -> Compound:
    data = input.df.open(pd.DataFrame).all()
    dk = Deck("Data")
    dk.append(data.to_html())
    return input


@workflow
def wf() -> Compound:
    md = make_data()
    ret = pass_data(md)
    return ret


if __name__ == "__main__":
    md = make_data()
    ret = pass_data(md)
