import sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
# from flytekitplugins.flyteinteractive import vscode
from flytekit import ImageSpec
from pathlib import Path
from flytekit import task, workflow


image = ImageSpec(
    requirements=Path(__file__).parent / "requirements.txt",
)

@task(  
    container_image=image,
)
def load_data() -> pd.DataFrame:
    # Load dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = pd.Series(data.target)
    return df

@task(  
    container_image=image,
)
def preprocess_data(test_size: float, data: pd.DataFrame) ->\
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    labels = data['target']
    features = data.drop('target', axis=1)
    # Split our data
    train, test, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=42)
    return train, test, train_labels, test_labels


@task(
    container_image=image,
)
def get_result(
        train: pd.DataFrame, test: pd.DataFrame,
        train_labels: pd.Series, test_labels: pd.Series
        ) -> float:

    print(train.shape, test.shape, train_labels.shape, test_labels.shape)
    return .5

@workflow
def training_model_wf(test_size: float=.33):
    print(f"test_size={test_size}")
    data = load_data()
    pp_data = preprocess_data(test_size, data)
    # print(pp_data.shape)
    return get_result(pp_data[0], pp_data[1], pp_data[2], pp_data[3])

# training_model_wf()
