import torch

from scripts.eval.compare_qwen_latent_gaussian_norms import (
    resolve_layer_index,
    summarize_norms,
    update_running_moments,
)


def test_final_layer_input_index_for_qwen_28_layers():
    assert resolve_layer_index(-1, 28) == 27


def test_norm_summary_reports_vector_dimension_and_rms():
    vectors = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    summary = summarize_norms(vectors)
    assert summary["num_vectors"] == 2
    assert summary["dimension"] == 2
    assert summary["l2_mean"] == 5.0
    assert torch.isclose(torch.tensor(summary["rms_mean"]), torch.tensor(5 / 2**0.5))


def test_streaming_moments_match_full_dataset_statistics():
    first = torch.tensor([[1.0, 8.0], [3.0, 4.0]])
    second = torch.tensor([[5.0, 0.0], [7.0, -4.0], [9.0, -8.0]])
    mean = torch.zeros(2, dtype=torch.float64)
    m2 = torch.zeros(2, dtype=torch.float64)

    count, mean, m2 = update_running_moments(0, mean, m2, first)
    count, mean, m2 = update_running_moments(count, mean, m2, second)

    combined = torch.cat([first, second]).double()
    assert count == combined.shape[0]
    assert torch.allclose(mean, combined.mean(dim=0))
    assert torch.allclose((m2 / (count - 1)).sqrt(), combined.std(dim=0))
