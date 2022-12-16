from flask import Flask, render_template, url_for, request, redirect, flash
import stability_mpea as sm
from markupsafe import escape

app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def index():
    status = True
    if request.method == "POST":
        formula = request.form['formula']
        t_fac = request.form['t_fac']

        if formula:
            status = sm.sane(escape(formula))
            if status:
                return redirect(url_for('results', formula=formula, t_fac=t_fac))

    return render_template('index.html', status=status)

@app.route('/<formula>:<t_fac>')
def results(formula, t_fac):
    t_fac = float(escape(t_fac))
    res = sm.predict(escape(formula), t_fac)
    return render_template('results.html', res=res, t_fac=t_fac)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/<formula>')
def stability(formula):
    t_fac = 0.9
    res = sm.predict(escape(formula), t_fac)
    msg = '<h2>{0}</h2><h3>Stability at {1}Tm ({2:.0f} K): {3}</h3><h3>Melting point (Tm, rule of mixtures): {4:.0f} K</h3>'.format(res[0]['system'], t_fac, res[0]['t'], res[0]['stability'], res[0]['tm'])
    if res[0]['stability'] == 'unstable':
        msg_2 = '<h3>Energy above hull: {0:.3f} eV</h3><h3>Predicted microstructures: {1}</h3>'.format(res[0]['e_above_im'], res[0]['decomp'])
    else:
        msg_2 = '<h3>Inverse energy above hull: {0:.3f} eV</h3><h3>Predicted structure: {1}</h3>'.format(res[0]['e_above_im'], res[0]['phase'])
    return msg+msg_2
