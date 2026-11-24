import unittest
from xml.etree import ElementTree

from flask import render_template

from flask_app import app
from stability_mpea import init_params, predict


class HeatingExampleTest(unittest.TestCase):
    def test_hull_markers_match_the_bundled_formation_energies(self):
        with app.test_request_context():
            figure = ElementTree.fromstring(render_template("_feco_hull.html"))
        dots = figure.findall(".//g[@id='fixed-phases']/circle")

        params, compounds, _ = init_params()
        references = {el: min(lattice[el] for lattice in params["elements"].values())
                      for el in ("Fe", "Co")}
        lowest = {0: 0, 1: 0}
        for entry in compounds["Co-Fe"]:
            counts = entry["unit_cell_formula"]
            atoms = sum(counts.values())
            x = counts["Co"] / atoms
            reference = sum(count * references[el] for el, count in counts.items())
            formation = 1000 * (entry["total_energy"] - reference) / atoms
            lowest[x] = min(lowest.get(x, formation), formation)

        self.assertEqual(len(dots), len(lowest))
        for dot, (x, energy) in zip(dots, sorted(lowest.items())):
            self.assertAlmostEqual(float(dot.attrib["data-x"]), x)
            self.assertAlmostEqual(float(dot.attrib["data-energy"]), energy)

    def test_illustrated_crossover_agrees_with_full_phase_competition(self):
        with app.test_request_context():
            figure = ElementTree.fromstring(render_template("_feco_plot.html"))
        data = figure.find("svg").attrib
        enthalpy = float(data["data-enthalpy"])
        entropy = float(data["data-entropy"])
        melting = float(data["data-melting"])
        cold, hot = int(enthalpy / entropy), int(enthalpy / entropy) + 1

        for temperature in (0, cold, hot, 1500):
            with self.subTest(temperature=temperature):
                result, _, _ = predict("FeCo", temperature / melting)[0]
                self.assertAlmostEqual(result["t"], temperature)
                self.assertAlmostEqual(-1000 * result["ts_conf"], temperature * entropy)
                self.assertEqual(result["phase"], "BCC")
                if temperature <= cold:
                    self.assertEqual(result["stability"], "unstable")
                    self.assertEqual(result["decomp"], ["FeCo(B2)"])
                    self.assertAlmostEqual(1000 * result["e_above_im"], enthalpy - temperature * entropy)
                else:
                    self.assertEqual(result["stability"], "stable")
