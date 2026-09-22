import numpy as np
from sklearn.datasets import fetch_openml

def try_load(label, **kw):
    try:
        data = fetch_openml(**kw)
        if hasattr(data, "data"):
            X, y = data.data, data.target
        else:
            X, y = data
        X = np.asarray(X)
        y = np.asarray(y)
        print(label, "OK", "X", getattr(X, "shape", type(X)), "y", getattr(y, "shape", type(y)))
    except Exception as e:
        print(label, "FAIL", type(e).__name__, str(e)[:120])

try_load("energy name", name="energy_efficiency", version=1, as_frame=True, return_X_y=True)
try_load("energy id128 frame", data_id=128, as_frame=True, return_X_y=True)
try_load("concrete Concrete", name="Concrete", version=1, as_frame=False, return_X_y=True)
try_load("concrete id187", data_id=187, as_frame=False, return_X_y=True)
try_load("protein id42730", data_id=42730, as_frame=False, return_X_y=True)
try_load("year id218 frame", data_id=218, as_frame=True, return_X_y=True)
try_load("cpu small", data_id=228, as_frame=False, return_X_y=True)
