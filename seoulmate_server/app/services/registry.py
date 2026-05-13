from app.services import (
    safety_service,
    health_service,
    comfort_service,
    stress_service,
    hvac_service,
    expenses_service,
)


LAYER_SERVICES = {
    "safety": safety_service,
    "health": health_service,
    "comfort": comfort_service,
    "stress": stress_service,
    "hvac": hvac_service,
    "expenses": expenses_service,
}


def get_service(layer: str):
    if layer not in LAYER_SERVICES and layer != "overall":
        return None

    return LAYER_SERVICES.get(layer)