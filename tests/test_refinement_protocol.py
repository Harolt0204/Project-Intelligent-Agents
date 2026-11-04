"""Evitar una tasa incorrecta o una comparación desigual en ejecuciones largas."""

import json
from pathlib import Path
import unittest

from experiments.refinar_lunar import learning_rate, job_for


class RefinementProtocolTests(unittest.TestCase):
    def test_schedule_and_comparison_are_fixed_before_unseen_tests(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/tier23_mejora_v1.json').read_text(encoding='utf-8'))
        self.assertEqual(config['seeds'],[42,43,44])
        self.assertEqual(job_for(config,'base')['parameters'],job_for(config,'double')['parameters'])
        self.assertAlmostEqual(learning_rate(0,config),0.00063)
        self.assertAlmostEqual(learning_rate(300000,config),0.00063)
        self.assertAlmostEqual(learning_rate(550000,config),0.00034)
        self.assertAlmostEqual(learning_rate(800000,config),0.00005)
        self.assertAlmostEqual(learning_rate(1000000,config),0.00005)
        self.assertEqual(len({config['evaluation_base'],config['heldout_evaluation_base'],config['confirmation_evaluation_base']}),3)


if __name__=='__main__': unittest.main()
