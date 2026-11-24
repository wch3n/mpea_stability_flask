from functools import lru_cache

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

import stability_mpea as sm

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


def prediction_inputs(formula, t_fac):
    formula = formula.strip()
    if not sm.sane(formula):
        raise ValueError("Enter a supported alloy formula with 2 to 8 elements.")
    try:
        factor = float(t_fac)
    except (TypeError, ValueError):
        factor = None
    if factor is None or not 0 <= factor <= 1:
        raise ValueError("Enter a temperature factor between 0 and 1.")
    return formula, factor


@lru_cache(maxsize=128)
def prediction(formula, factor):
    return sm.predict(formula, factor)


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        try:
            formula, factor = prediction_inputs(request.form.get("formula", ""), request.form.get("t_fac"))
        except ValueError as exc:
            return render_template("index.html", error=str(exc)), 400
        return redirect(url_for("results", formula=formula, t_fac=factor))
    return render_template("index.html")


@app.route("/<formula>:<t_fac>")
def results(formula, t_fac):
    try:
        formula, factor = prediction_inputs(formula, t_fac)
    except ValueError as exc:
        abort(400, description=str(exc))
    return render_template("results.html", res=prediction(formula, factor), t_fac=factor)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/<formula>:<t_fac>")
def stability(formula, t_fac):
    try:
        formula, factor = prediction_inputs(formula, t_fac)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(prediction(formula, factor)[0][0])
