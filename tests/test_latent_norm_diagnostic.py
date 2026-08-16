import torch

from scripts.eval.compare_qwen_latent_gaussian_norms import (
    resolve_layer_index,
    summarize_norms,
)


def test_final_layer_input_index_for_qwen_28_layers():
    assert resolve_layer_index(-1, 28) == 27


def test_norm_summary_reports_vector_dimension_and_rms():
    vectors = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    summary = summarize_norms(vectors)
    assert summary["num_vectors"] == 2
    assert summary["dimension"] == 2
    assert summary["l2_mean"] == 5.0
    assert torch.isclose(
        torch.tensor(summary["rms_mean"]), torch.tensor(5 / 2**0.5)
    )
