import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from flytekit import ImageSpec
from pathlib import Path
from flytekit import task, workflow, Deck
import matplotlib as mpl
import io, base64


# Settings for the workflow
cache_enabled = False
cache_version = "1.1"
data_url = 'https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv'
add_regressors = True
playoffs = pd.DataFrame({
    'holiday': 'playoff',
    'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
    'lower_window': 0,
    'upper_window': 1,
})

image = ImageSpec(
    requirements=Path(__file__).parent / "requirements.txt",
)


@task(
    container_image=image,
    cache=cache_enabled,
    cache_version=cache_version,
)
def load_ts_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df


@task(
    container_image=image,
    cache=cache_enabled,
    cache_version=cache_version,
)
def train_model(df: pd.DataFrame) -> Prophet:
    if add_regressors:
        model = Prophet(holidays=playoffs)
    else:
        model = Prophet()
    model.fit(df)
    return model


@task(
    container_image=image,
    cache=cache_enabled,
    cache_version=cache_version,
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


@task(
    container_image=image,
    enable_deck=True,
)
def evaluate_model(model: Prophet) -> pd.DataFrame:
    dk = Deck("Model Evaluation")
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
    fig1 = plot_cross_validation_metric(df_cv, metric='mape')
    dk.append(_convert_fig_into_html(fig1))
    df_p = performance_metrics(df_cv)
    dk.append(df_p.to_html())
    return df_p


@workflow
def prophet_workflow(url: str) -> pd.DataFrame:
    url = data_url

    df = load_ts_data(url)
    model = train_model(df)
    forecast = predict(model, df)
    make_plots(model, forecast)
    evaluate_model(model)
    return forecast


def _convert_fig_into_html(fig: mpl.figure.Figure) -> str:
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    img_base64 = base64.b64encode(img_buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_base64}" alt="Rendered Image" />'

if __name__ == "__main__":
    print("In Main")
    prophet_workflow("")