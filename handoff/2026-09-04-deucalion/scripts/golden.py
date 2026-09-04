"""Bit-exact golden dump of single-partition propagation for A/B comparison of engine arms.

Usage: python golden.py <repo-root> <out.json>   (run with the arm's venv python, monoprop_PARTITIONS=1)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("monoprop_PARTITIONS", "1")
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
import numpy as np  # noqa: E402

import monoprop  # noqa: E402
from monoprop import MajoranaPropagator  # noqa: E402
from tests.cases import load_problem  # noqa: E402

fixtures = sorted((root / "tests" / "data").glob("*.msgpack"))
configs = [
    dict(name="c4", cutoff=4),
    dict(name="c6_la1e-6", cutoff=6, lower_atol=1e-6),
    dict(name="c4_la1e-4_ua1e-3", cutoff=4, lower_atol=1e-4, upper_atol=1e-3),
    dict(name="c6_schro", cutoff=6, schrodinger_cutoff=6),
    dict(name="c4_schro_la1e-5", cutoff=4, schrodinger_cutoff=4, lower_atol=1e-5),
]
out = {"core_md5": None, "cases": {}}
try:
    import hashlib
    so = Path(monoprop.__file__).parent / "_core.abi3.so"
    out["core_md5"] = hashlib.md5(so.read_bytes()).hexdigest()
except Exception as exc:  # noqa: BLE001
    out["core_md5"] = f"unavailable: {exc}"

for fx in fixtures:
    prob = load_problem(fx)
    circuit = prob.monomial_circuit.to_circuit()
    for cfg in configs:
        kw = {k: v for k, v in cfg.items() if k != "name"}
        rec = {}
        for mode in ("propagate", "graph"):
            mp = MajoranaPropagator(prob.operator, circuit.initial_state, **kw)
            if mode == "propagate":
                mp.propagate(circuit)
                ev = mp.expectation_value()
                rec[mode] = {"size": mp.size(), "expval": repr(float(ev))}
                terms = mp.evolved_operator(atol=0.0).terms
                rec[mode]["terms"] = sorted(
                    (list(k), repr(complex(v).real), repr(complex(v).imag)) for k, v in terms.items()
                )
            else:
                mp.build_graph(circuit)
                ev, grad = mp.expectation_value_and_gradient(np.asarray(prob.monomial_circuit.parameters))
                rec[mode] = {
                    "size": mp.size(),
                    "graph_size": list(mp.graph_size()),
                    "expval": repr(float(ev)),
                    "grad": [repr(float(g)) for g in np.asarray(grad)],
                }
        out["cases"][f"{fx.stem}/{cfg['name']}"] = rec
        print(fx.stem, cfg["name"], rec["propagate"]["size"], rec["propagate"]["expval"], rec["graph"]["expval"], flush=True)

Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print("wrote", sys.argv[2], out["core_md5"])
