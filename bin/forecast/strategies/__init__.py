from .beam_search import BeamSearch
forecast_strategies = {
    "beam_search": BeamSearch,
}

def find_forecast_strategy(strategy):
    if strategy not in forecast_strategies:
        raise ValueError("Decode strategy {} did not exist in our system".\
                format(strategy))
    return forecast_strategies[strategy]