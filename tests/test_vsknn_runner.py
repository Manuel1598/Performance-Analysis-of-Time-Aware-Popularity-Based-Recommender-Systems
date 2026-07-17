import unittest

from src.recbole_framework.runners.session.run_vsknn_recbole import (
    SAMPLE_DATASETS,
    SESSION_DATASETS,
    parse_args,
)


class VSKNNRunnerTests(unittest.TestCase):
    def test_default_dataset_is_yoochoose_sample(self):
        args = parse_args([])
        self.assertEqual(args.dataset, "yoochoose_recbole_sample")
        self.assertFalse(args.all_samples)

    def test_each_session_dataset_is_accepted(self):
        for dataset in SESSION_DATASETS:
            with self.subTest(dataset=dataset):
                args = parse_args(["--dataset", dataset])
                self.assertEqual(args.dataset, dataset)

    def test_all_samples_contains_three_domains(self):
        self.assertEqual(
            SAMPLE_DATASETS,
            (
                "yoochoose_recbole_sample",
                "globo_recbole_sample",
                "adressa_recbole_sample",
            ),
        )

    def test_dataset_and_all_samples_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            parse_args(
                ["--dataset", "globo_recbole_sample", "--all-samples"]
            )


if __name__ == "__main__":
    unittest.main()
