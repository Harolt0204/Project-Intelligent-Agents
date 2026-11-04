"""Contrato experimental: cambiar una variable y reutilizar un único baseline."""

import unittest

from experiments.lunarlander import configuration,plan


class PlanTests(unittest.TestCase):
    def test_seven_groups_cover_two_three_value_studies_and_matched_extensions(self):
        config=configuration()
        jobs=plan(config)
        self.assertEqual(len(jobs),7)
        self.assertEqual(len(config['seeds']),3)
        for key,values in config['studies'].items():
            self.assertEqual(len(set(values)),3)
            self.assertIn(config['base'][key],values)
        for name,job in jobs.items():
            changed=[k for k in config['base'] if job['parameters'][k]!=config['base'][k]]
            self.assertEqual(changed,[job['role']] if job['role'] in config['studies'] else [])
        self.assertEqual(jobs['double']['parameters'],jobs['base']['parameters'])
        self.assertEqual(jobs['no_target']['parameters'],jobs['base']['parameters'])


if __name__=='__main__': unittest.main()
