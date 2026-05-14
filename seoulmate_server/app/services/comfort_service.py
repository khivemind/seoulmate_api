from app.services.api_result_loader import load_layer_result


def get_heatmap(year: int, month: int):
    return load_layer_result("comfort", year, month)