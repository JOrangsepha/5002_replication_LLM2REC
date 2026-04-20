from .SASRec._model import SASRec
from .GRU4Rec._model import GRU4Rec

MODEL_REGISTRY = {
    'SASRec': SASRec,
    'GRU4Rec': GRU4Rec,
}
