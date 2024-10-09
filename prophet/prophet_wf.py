import pandas as pd
from prophet import Prophet
from flytekit import ImageSpec
from pathlib import Path
from flytekit import task, workflow, Deck
import matplotlib as mpl
import io, html, base64


image = ImageSpec(
    requirements=Path(__file__).parent / "requirements.txt",
)


@task(
    container_image=image,
    cache_version="1.0",
)
def load_ts_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df


@task(
    container_image=image,
    cache_version="1.0",
)
def train_model(df: pd.DataFrame) -> Prophet:
    model = Prophet()
    model.fit(df)
    return model


@task(
    container_image=image,
    cache_version="1.0",
)
def predict(model: Prophet, df: pd.DataFrame) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast


@task(
    container_image=image,
    enable_deck=True,
)
def make_plots(model: Prophet, forecast: pd.DataFrame):
    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)

    dk = Deck("Prophet")
    dk.append(_convert_fig_into_html(fig1))
    dk.append(_convert_fig_into_html(fig2))


@workflow
def prophet_workflow(url: str) -> pd.DataFrame:
    url = 'https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv'

    df = load_ts_data(url)
    model = train_model(df)
    forecast = predict(model, df)
    make_plots(model, forecast)
    return forecast


def _convert_fig_into_html(fig: mpl.figure.Figure) -> str:
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    img_base64 = base64.b64encode(img_buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_base64}" alt="Rendered Image" />'

if __name__ == "__main__":
    print("In Main")
    prophet_workflow("")