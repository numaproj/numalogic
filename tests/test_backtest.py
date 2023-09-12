import os.path
import shutil
import unittest
from unittest.mock import patch, Mock

import pandas as pd
from typer.testing import CliRunner
from numalogic.backtest._bt import app as btapp
from numalogic.backtest.__main__ import app as rootapp
from numalogic._constants import TESTS_DIR
from numalogic.connectors import PrometheusFetcher
from numalogic.tools.exceptions import DataFormatError

runner = CliRunner()


def _mock_datafetch():
    df = pd.read_csv(
        os.path.join(TESTS_DIR, "resources", "data", "interactionstatus.csv"),
        usecols=["ts", "failure"],
        nrows=1000,
    )
    df.rename(columns={"ts": "timestamp"}, inplace=True, errors="raise")
    return df


class TestBacktest(unittest.TestCase):
    def tearDown(self) -> None:
        _dir = os.path.join(TESTS_DIR, ".btoutput")
        if os.path.exists(_dir):
            shutil.rmtree(_dir, ignore_errors=False, onerror=None)

    @patch.object(PrometheusFetcher, "fetch", Mock(return_value=_mock_datafetch()))
    def test_univar(self):
        res = runner.invoke(btapp, ["univariate", "ns1", "app1", "failure"], catch_exceptions=False)
        self.assertEqual(0, res.exit_code)
        self.assertIsNone(res.exception)

    def test_multivar(self):
        with self.assertRaises(NotImplementedError):
            runner.invoke(btapp, ["multivariate"], catch_exceptions=False)


class TestRoot(unittest.TestCase):
    def tearDown(self) -> None:
        _dir = os.path.join(TESTS_DIR, ".btoutput")
        if os.path.exists(_dir):
            shutil.rmtree(_dir, ignore_errors=False, onerror=None)

    def test_train(self):
        data_path = os.path.join(TESTS_DIR, "resources", "data", "interactionstatus.csv")
        res = runner.invoke(
            rootapp,
            [
                "train",
                "--data-file",
                data_path,
                "--col-name",
                "failure",
                "--ts-col-name",
                "ts",
                "--train-ratio",
                "0.1",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(0, res.exit_code)
        self.assertIsNone(res.exception)

    def test_train_err(self):
        data_path = os.path.join(TESTS_DIR, "resources", "data", "interactionstatus.csv")
        with self.assertRaises(DataFormatError):
            runner.invoke(
                rootapp,
                [
                    "train",
                    "--data-file",
                    data_path,
                    "--col-name",
                    "failure",
                    "--ts-col-name",
                    "timestamp",
                    "--train-ratio",
                    "0.1",
                ],
                catch_exceptions=False,
            )

    def test_train_arg_err(self):
        res = runner.invoke(
            rootapp,
            [
                "train",
                "--col-name",
                "failure",
                "--ts-col-name",
                "timestamp",
                "--train-ratio",
                "0.1",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(1, res.exit_code)
        self.assertIsNotNone(res.exception)

    def test_score(self):
        data_path = os.path.join(TESTS_DIR, "resources", "data", "interactionstatus.csv")
        res = runner.invoke(
            rootapp,
            [
                "score",
                "--data-file",
                data_path,
                "--col-name",
                "failure",
                "--ts-col-name",
                "ts",
                "--test-ratio",
                "0.5",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(0, res.exit_code)
        self.assertIsNone(res.exception)

    def test_score_err(self):
        data_path = os.path.join(TESTS_DIR, "resources", "data", "interactionstatus.csv")
        with self.assertRaises(DataFormatError):
            runner.invoke(
                rootapp,
                [
                    "score",
                    "--data-file",
                    data_path,
                    "--col-name",
                    "failure",
                    "--ts-col-name",
                    "timestamp",
                    "--test-ratio",
                    "0.1",
                ],
                catch_exceptions=False,
            )

    def test_score_arg_err(self):
        res = runner.invoke(
            rootapp,
            [
                "score",
                "--col-name",
                "failure",
                "--ts-col-name",
                "timestamp",
                "--test-ratio",
                "0.1",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(1, res.exit_code)
        self.assertIsNotNone(res.exception)

    def test_clear(self):
        res = runner.invoke(
            rootapp,
            ["clear", "--output-dir", os.path.join(TESTS_DIR, ".btoutput"), "--all"],
            catch_exceptions=False,
        )
        self.assertEqual(0, res.exit_code)
        self.assertIsNone(res.exception)


if __name__ == "__main__":
    unittest.main()
