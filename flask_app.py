from flask import Flask, render_template, url_for, request, redirect, flash
import stability_mpea as sm
from markupsafe import escape

app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def index():
    status = True
    if request.method == "POST":
        formula = escape(request.form['formula']).lstrip()
        t_fac = escape(request.form['t_fac'])

        if formula:
            status = sm.sane(formula)
            if status:
                return redirect(url_for('results', formula=formula, t_fac=t_fac))

    return render_template('index.html', status=status)

@app.route('/<formula>:<t_fac>')
def results(formula, t_fac):
    t_fac = float(t_fac)
    res = sm.predict(formula, t_fac)
    return render_template('results.html', res=res, t_fac=t_fac)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/<formula>:<t_fac>')
def stability(formula,t_fac):
    import json
    status = sm.sane(escape(formula))
    if status:
        res = sm.predict(escape(formula), float(escape(t_fac)))
        return json.dumps(res[0][0])
    else:
        return "invalid formula"
