import numpy as np
import pandas as pd
import torch

from cancer_detection import data, fusion, multitask, tabular


def metadata(n_patients=60, seed=0):
    """Synthetic train.csv: two images per patient, about one patient in six with cancer."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        cancer = int(p % 6 == 0)
        for view in range(2):
            rows.append({
                "patient_id": p, "image_id": p * 10 + view, "site_id": 1, "implant": 0,
                "age": float(rng.integers(40, 80)) if p % 13 else np.nan,
                "density": "ABCD"[p % 4], "cancer": cancer,
            })
    return pd.DataFrame(rows)


def test_clean_drops_missing_and_encodes_density():
    df = data.clean(metadata())
    assert df["age"].notna().all() and "site_id" not in df and "implant" not in df
    assert set(df["density"]) <= {0, 1, 2, 3}


def test_balance_by_image_is_exact_and_has_no_duplicates():
    df = data.clean(metadata())
    b = data.balance_by_image(df)
    assert b["cancer"].sum() * 2 == len(b)
    assert b["image_id"].is_unique


def test_balance_by_patient_keeps_whole_patients():
    df = data.clean(metadata())
    b = data.balance_by_patient(df)
    per_patient = b.groupby("patient_id")["cancer"].max()
    assert per_patient.sum() * 2 == len(per_patient)
    assert (b.groupby("patient_id").size() == df[df["patient_id"].isin(b["patient_id"])]
            .groupby("patient_id").size()).all()


def test_patient_split_has_no_leakage():
    train, test = data.patient_split(data.clean(metadata()), test_size=0.2)
    assert not set(train["patient_id"]) & set(test["patient_id"])


def test_tabular_baseline_fits_and_predicts():
    df = data.clean(metadata())
    train, test = data.patient_split(df, test_size=0.2)
    for normalise in (False, True):
        pred = tabular.fit(train, normalise).predict(test[tabular.FEATURES])
        assert set(pred) <= {0, 1}


def test_fusion_joins_on_image_id_not_position():
    df = data.clean(metadata())
    ids = df["image_id"].to_numpy()[::-1]  # deliberately not in df order
    emb = ids[:, None].astype(float)  # each embedding is its own image id
    fused = fusion.fusion_frame(ids, emb, df)
    assert (fused["f0"].to_numpy() == fused.index.to_numpy()).all()
    assert (fused["cancer"] == df.set_index("image_id").loc[fused.index, "cancer"]).all()


def test_age_encoding_round_trips():
    age = torch.tensor([40.0, 65.0, 89.0])
    assert torch.allclose(multitask.decode_age(multitask.encode_age(age)), age)
    assert multitask.encode_age(age).max() <= 1


def test_multitask_model_trains_one_step():
    torch.manual_seed(0)
    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 16))
    model = multitask.MultiTaskModel(encoder, n_features=16)
    loss_fn = multitask.UncertaintyWeightedLoss()
    x = torch.randn(4, 3, 8, 8)
    y = (multitask.encode_age(torch.tensor([50.0, 60, 70, 80])),
         torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 2, 3]))
    age, cancer, density = model(x)
    assert age.shape == (4,) and cancer.shape == (4, 2) and density.shape == (4, 4)
    loss = loss_fn((age, cancer, density), y)
    loss.backward()
    assert loss_fn.log_vars.grad is not None


def test_f1_cancer_predicts_class_one_from_its_logit():
    logits = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
    assert multitask.f1_cancer(logits, torch.tensor([1, 0])) == 1.0
