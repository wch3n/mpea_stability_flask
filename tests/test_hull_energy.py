import math
import unittest

from stability_mpea import K, predict


class HullEnergyTest(unittest.TestCase):
    def test_alco_reports_energy_of_predicted_polymorph(self):
        result, _, _ = predict("AlCo", 0.9)[0]
        self.assertEqual(result["phase"], "HCP")
        self.assertEqual(result["stability"], "unstable")
        self.assertEqual(result["decomp"], ["AlCo(B2)"])

        # Equiatomic HCP references and interaction in assets/omegas.json.
        hcp = (-3.71740875 - 7.10931223) / 2 - 1.0999706975000016 / 4
        hcp -= K * result["t"] * math.log(2)
        # Lowest B2 cell energy in im_aflow_icsd.json, divided by two atoms.
        self.assertAlmostEqual(result["e_above_im"], hcp - (-12.07863178 / 2))
