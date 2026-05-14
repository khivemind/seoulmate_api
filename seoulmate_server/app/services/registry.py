from app.services import safety_service
from app.services import comfort_service
from app.services import health_service
from app.services import stress_service
from app.services import hvac_service
from app.services import expenses_service


LAYER_SERVICES = {
    "safety": safety_service,
    "comfort": comfort_service,
    "health": health_service,
    "stress": stress_service,
    "hvac": hvac_service,
    "expenses": expenses_service,
}


def get_service(layer: str):
    if layer not in LAYER_SERVICES and layer != "overall":
        return None

    return LAYER_SERVICES.get(layer)