"""Middle fusion: image embeddings joined with the patient metadata."""

import numpy as np
import pandas as pd


def fusion_frame(image_ids, embeddings: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """One row per image: its metadata (age, density, cancer) plus its embedding.

    Rows are joined on image_id. The notebook paired them by position instead, while the
    training embeddings came from a shuffled dataloader and the metadata from df.iloc on
    indices of the image-file list, so embeddings and metadata did not line up.
    """
    emb = pd.DataFrame(np.asarray(embeddings), index=pd.Index(image_ids, name="image_id"))
    emb.columns = [f"f{i}" for i in emb.columns]
    meta = df.set_index("image_id")[["age", "density", "cancer"]]
    return meta.join(emb, how="inner")
