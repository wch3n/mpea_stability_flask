# MPEA stability

A Flask app for predicting whether a multi-principal element alloy forms a
single disordered solid solution at a specified fraction of its estimated
melting point. Try the [live predictor](https://mpea.alloys.dev/) or explore
the [model equations and interactive FeCo heating example](https://mpea.alloys.dev/about).

## Run locally

From the repository root, with Python 3.11 or newer:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m flask --app flask_app run
```

Open [localhost:5000](http://localhost:5000/). The API accepts a formula
containing 2–8 supported elements and a temperature factor from 0 to 1.
For example, `curl http://localhost:5000/api/AlCo:0.9` returns the HCP
solid solution as unstable, about 0.27835 eV/atom above the hull.

Run the regression test with `python -m unittest discover -s tests`.
See the [deployment guide](deploy/README.md) for Gunicorn, Lighttpd and HTTPS.

The model follows W. Chen et al., “A map of single-phase high-entropy alloys,”
*Nature Communications* **14**, 2856 (2023).
[Paper](https://doi.org/10.1038/s41467-023-38423-7) · [Wei Chen’s website](https://wch3n.github.io/)
