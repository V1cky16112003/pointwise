"""Finding run 3's weights among whatever Kaggle mounts under /kaggle/input."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts import kaggle_job as K
from scripts.phase5_eval import RUN3_BEST_STEP


@pytest.fixture
def mount(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "Path", lambda _: tmp_path)  # stands in for Path("/kaggle/input")
    return tmp_path


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    return path


def test_finds_a_flat_dataset_upload_and_skips_other_files(mount):
    save(mount / "other" / "state.pt", {"step": 834})  # another run's checkpoint
    save(mount / "junk" / "state.pt", {"not": "a checkpoint"})
    want = save(mount / "datasets" / "pointwise-run3-weights" / "state.pt", {"step": RUN3_BEST_STEP})
    assert K.find_run3_state() == want


def test_exits_with_what_it_found_when_no_run3_weights(mount):
    save(mount / "a" / "state.pt", {"step": 834})
    with pytest.raises(SystemExit, match="found .*834"):
        K.find_run3_state()
