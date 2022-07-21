import os
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

from numalogic._constants import TESTS_DIR
from numalogic.models.autoencoder import AutoencoderPipeline, SparseAEPipeline
from numalogic.models.autoencoder.variants import Conv1dAE, LSTMAE, VanillaAE, TransformerAE

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")
torch.manual_seed(42)
SEQ_LEN = 12


class TestAutoEncoderPipeline(unittest.TestCase):
    model = None
    X_train = None
    X_val = None

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(DATA_FILE)
        df = df[["success", "failure"]]
        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(df[:-240])
        cls.X_val = scaler.transform(df[-240:])

    def test_fit_conv(self):
        self.model = Conv1dAE(self.X_train.shape[1], 8)
        trainer = AutoencoderPipeline(self.model, SEQ_LEN, num_epochs=5)
        trainer.fit(self.X_train)

    def test_fit_lstm(self):
        self.model = LSTMAE(seq_len=SEQ_LEN, no_features=2, embedding_dim=16)
        trainer = AutoencoderPipeline(self.model, SEQ_LEN, num_epochs=5)
        trainer.fit(self.X_train)

    def test_fit_vanilla(self):
        self.model = VanillaAE(SEQ_LEN, n_features=self.X_train.shape[1])
        trainer = AutoencoderPipeline(self.model, SEQ_LEN, num_epochs=5)
        trainer.fit(self.X_train)

    def test_threshold_min(self):
        self.model = Conv1dAE(self.X_train.shape[1], 8)
        trainer = AutoencoderPipeline(self.model, SEQ_LEN, num_epochs=5, threshold_min=1)
        trainer.fit(self.X_train)
        self.assertTrue(all(i >= 1 for i in trainer.thresholds))

    def test_predict_01(self):
        self.model = Conv1dAE(self.X_train.shape[1], 8)
        trainer = AutoencoderPipeline(self.model, SEQ_LEN, num_epochs=5, loss_fn="l1")
        pipeline = make_pipeline(StandardScaler(), trainer)
        pipeline.fit(self.X_train)
        pred = pipeline.predict(self.X_val, seq_len=12)

        self.assertEqual(self.X_val.shape, pred.shape)

    def test_predict_as_pl(self):
        pipeline = make_pipeline(
            RobustScaler(),
            AutoencoderPipeline(
                Conv1dAE(self.X_train.shape[1], 8), SEQ_LEN, num_epochs=5, loss_fn="mse"
            ),
        )
        pipeline.fit(self.X_train)
        pred = pipeline.predict(self.X_val, seq_len=12)
        self.assertEqual(self.X_val.shape, pred.shape)

    def test_predict_02(self):
        stream_data = self.X_val[:12]
        trainer = AutoencoderPipeline.with_model(
            VanillaAE, SEQ_LEN, num_epochs=5, signal_len=SEQ_LEN, n_features=self.X_train.shape[1]
        )
        trainer.fit(self.X_train)
        pred = trainer.predict(stream_data)

        self.assertEqual(stream_data.shape, pred.shape)

    def test_fit_predict(self):
        trainer = AutoencoderPipeline.with_model(
            VanillaAE, SEQ_LEN, num_epochs=5, signal_len=SEQ_LEN, n_features=self.X_train.shape[1]
        )
        pred = trainer.fit_predict(self.X_train)
        self.assertEqual(self.X_train.shape, pred.shape)

    def test_score_01(self):
        model = VanillaAE(SEQ_LEN, n_features=self.X_train.shape[1])
        trainer = AutoencoderPipeline(model, SEQ_LEN, num_epochs=5, optimizer="adagrad")
        trainer.fit(self.X_train)
        pred = trainer.predict(self.X_val)

        score = trainer.score(self.X_val)
        self.assertEqual(score.shape, pred.shape)
        self.assertEqual(trainer.reconerr_func, np.abs)

    def test_score_02(self):
        stream_data = self.X_val[:12]
        self.model = Conv1dAE(self.X_train.shape[1], 8)
        trainer = AutoencoderPipeline(self.model, SEQ_LEN, num_epochs=5, optimizer="rmsprop")
        trainer.fit(self.X_train)
        pred = trainer.predict(stream_data)

        score = trainer.score(stream_data)
        self.assertEqual(score.shape, pred.shape)

    def test_score_03(self):
        model = VanillaAE(SEQ_LEN, n_features=self.X_train.shape[1])
        trainer = AutoencoderPipeline(model, SEQ_LEN, num_epochs=5, reconerr_method="absolute")
        trainer.fit(self.X_train)
        pred = trainer.predict(self.X_val)

        score = trainer.score(self.X_val)
        self.assertEqual(score.shape, pred.shape)
        self.assertEqual(trainer.reconerr_func, np.abs)

    def test_score_04(self):
        model = TransformerAE(
            num_heads=8,
            seq_length=SEQ_LEN,
            dim_feedforward=64,
            num_encoder_layers=3,
            num_decoder_layers=1,
        )
        print(self.X_train.shape)
        trainer = AutoencoderPipeline(model, SEQ_LEN, num_epochs=5, reconerr_method="absolute")
        trainer.fit(self.X_train)
        pred = trainer.predict(self.X_val)

        score = trainer.score(self.X_val)
        self.assertEqual(score.shape, pred.shape)
        self.assertEqual(trainer.reconerr_func, np.abs)

    def test_score_05(self):
        model = VanillaAE(SEQ_LEN, n_features=self.X_train.shape[1])
        with self.assertRaises(ValueError):
            AutoencoderPipeline(model, SEQ_LEN, num_epochs=5, reconerr_method="noidea")

    def test_non_implemented_loss(self):
        model = VanillaAE(SEQ_LEN, n_features=self.X_train.shape[1])
        with self.assertRaises(Exception):
            AutoencoderPipeline(model, SEQ_LEN, num_epochs=5, loss_fn="lol")

    def test_non_implemented_optimizer(self):
        model = VanillaAE(SEQ_LEN, n_features=self.X_train.shape[1])
        with self.assertRaises(Exception):
            AutoencoderPipeline(model, SEQ_LEN, num_epochs=5, optimizer="lol")

    def test_save_load_path(self):
        path = "checkpoint.pth"

        try:
            os.remove(path)
        except OSError:
            pass

        self.model = Conv1dAE(self.X_train.shape[1], 8)
        trainer_1 = AutoencoderPipeline(self.model, SEQ_LEN, num_epochs=5)
        trainer_1.fit(self.X_train)
        trainer_1.save(path)

        trainer_2 = AutoencoderPipeline(self.model, SEQ_LEN, num_epochs=5)
        trainer_2.load(path)
        self.assertListEqual(trainer_1.thresholds.tolist(), trainer_2.thresholds.tolist())
        self.assertListEqual(
            trainer_1.err_stats["mean"].tolist(), trainer_2.err_stats["mean"].tolist()
        )
        self.assertListEqual(
            trainer_1.err_stats["std"].tolist(), trainer_2.err_stats["std"].tolist()
        )

        # Check if both model's weights are equal
        _mean_wts_1, _mean_wts_2 = [], []
        with torch.no_grad():
            for _w in trainer_1.model.parameters():
                _mean_wts_1.append(torch.mean(_w).item())
            for _w in trainer_2.model.parameters():
                _mean_wts_2.append(torch.mean(_w).item())

        self.assertTrue(_mean_wts_1)
        self.assertAlmostEqual(_mean_wts_1, _mean_wts_2, places=6)

        os.remove(path)

    def test_save_load_buf(self):
        model = VanillaAE(SEQ_LEN, n_features=self.X_train.shape[1])
        trainer_1 = AutoencoderPipeline(model, SEQ_LEN, num_epochs=5)
        trainer_1.fit(self.X_train)
        buf = trainer_1.save()

        trainer_2 = AutoencoderPipeline(model, SEQ_LEN, num_epochs=3)
        trainer_2.load(buf)
        self.assertListEqual(trainer_1.thresholds.tolist(), trainer_2.thresholds.tolist())
        self.assertListEqual(
            trainer_1.err_stats["mean"].tolist(), trainer_2.err_stats["mean"].tolist()
        )
        self.assertListEqual(
            trainer_1.err_stats["std"].tolist(), trainer_2.err_stats["std"].tolist()
        )

        # Check if both model's weights are equal
        _mean_wts_1, _mean_wts_2 = [], []
        with torch.no_grad():
            for _w in trainer_1.model.parameters():
                _mean_wts_1.append(torch.mean(_w).item())
            for _w in trainer_2.model.parameters():
                _mean_wts_2.append(torch.mean(_w).item())

        self.assertTrue(_mean_wts_1)
        self.assertAlmostEqual(_mean_wts_1, _mean_wts_2, places=6)

    def test_with_conv1d_model(self):
        trainer = AutoencoderPipeline.with_model(
            Conv1dAE, SEQ_LEN, in_channels=self.X_train.shape[1], enc_channels=8
        )
        self.assertIsInstance(trainer.model, Conv1dAE)

    def test_with_transformer_model(self):
        trainer = AutoencoderPipeline.with_model(
            TransformerAE,
            SEQ_LEN,
            num_heads=8,
            seq_length=SEQ_LEN,
            dim_feedforward=64,
            num_encoder_layers=3,
            num_decoder_layers=1,
        )
        self.assertIsInstance(trainer.model, TransformerAE)

    def test_load_model(self):
        X = np.random.randn(10, 1)
        model = VanillaAE(10)
        model_pl1 = AutoencoderPipeline(model, 10)
        model_pl1.fit(X)
        model_pl2 = AutoencoderPipeline(model, 10)
        model_pl2.load(model=model_pl1.model, **model_pl1.model_properties)
        self.assertEqual(model_pl2.err_stats["std"], model_pl1.err_stats["std"])

    def test_exception_in_load_model(self):
        X = np.random.randn(10, 1)
        model = VanillaAE(10)
        model_pl1 = AutoencoderPipeline(model, 10)
        model_pl1.fit(X)
        model_pl2 = AutoencoderPipeline(model, 10)
        with self.assertRaises(ValueError):
            model_pl2.load(
                path="checkpoint.pth", model=model_pl1.model, **model_pl1.model_properties
            )
            self.assertEqual(model_pl2.err_stats["std"], model_pl1.err_stats["std"])


class TestSparseAEPipeline(unittest.TestCase):
    X_train = None
    X_val = None

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(DATA_FILE)
        df = df[["success", "failure"]]
        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(df[:-240])
        cls.X_val = scaler.transform(df[-240:])

    def test_fit_kl_divergence_01(self):
        model = VanillaAE(SEQ_LEN, n_features=self.X_train.shape[1])
        trainer = SparseAEPipeline(model=model, seq_len=SEQ_LEN, num_epochs=5, beta=1e-2, rho=0.01)
        trainer.fit(self.X_train)

    def test_fit_kl_divergence_02(self):
        model = LSTMAE(seq_len=SEQ_LEN, no_features=self.X_train.shape[1], embedding_dim=16)
        trainer = SparseAEPipeline(model=model, seq_len=SEQ_LEN, num_epochs=5, beta=1e-2, rho=0.01)
        trainer.fit(self.X_train)

    def test_fit_kl_divergence_03(self):
        model = Conv1dAE(self.X_train.shape[1], 8)
        trainer = SparseAEPipeline(model=model, seq_len=SEQ_LEN, num_epochs=5, beta=1e-2, rho=0.01)
        trainer.fit(self.X_train)

    def test_fit_kl_divergence_04(self):
        model = TransformerAE(
            num_heads=8,
            seq_length=SEQ_LEN,
            dim_feedforward=64,
            num_encoder_layers=3,
            num_decoder_layers=1,
        )
        trainer = SparseAEPipeline(model=model, seq_len=SEQ_LEN, num_epochs=5, beta=1e-2, rho=0.01)
        trainer.fit(self.X_train)
        pred = trainer.predict(self.X_val, seq_len=SEQ_LEN)
        self.assertEqual(self.X_val.shape, pred.shape)

    def test_predict_as_pl(self):
        pipeline = Pipeline(
            [
                ("scaler", RobustScaler()),
                (
                    "sparse_ae",
                    SparseAEPipeline(
                        model=Conv1dAE(self.X_train.shape[1], 8), seq_len=SEQ_LEN, num_epochs=5
                    ),
                ),
            ]
        )
        pipeline.fit(self.X_train, sparse_ae__log_freq=1)
        pred = pipeline.predict(self.X_val, seq_len=12)
        self.assertEqual(self.X_val.shape, pred.shape)

    def test_fit_L1(self):
        model = VanillaAE(SEQ_LEN, n_features=self.X_train.shape[1])
        trainer = SparseAEPipeline(model=model, seq_len=SEQ_LEN, num_epochs=5, method="L1")
        trainer.fit(self.X_train)

    def test_fit_L2(self):
        model = VanillaAE(SEQ_LEN, n_features=self.X_train.shape[1])
        trainer = SparseAEPipeline(model=model, seq_len=SEQ_LEN, num_epochs=5, method="L2")
        trainer.fit(self.X_train)

    def test_non_implemented_loss(self):
        model = VanillaAE(SEQ_LEN, n_features=self.X_train.shape[1])
        trainer = SparseAEPipeline(model=model, seq_len=SEQ_LEN, num_epochs=5, method="lol")
        with self.assertRaises(NotImplementedError):
            trainer.fit(self.X_train)


if __name__ == "__main__":
    unittest.main()
