import torch

from landauer_llm.model import AperiodicRNN, LandauerLanguageModel


def test_aperiodic_rnn_shapes() -> None:
    torch.manual_seed(0)
    rnn = AperiodicRNN(input_size=4, hidden_size=6, num_layers=3)

    x = torch.randn(2, 4)
    top_state, stacked_states = rnn(x)

    assert top_state.shape == (2, 6)
    assert stacked_states.shape == (3, 2, 6)


def test_language_model_forward_and_generate() -> None:
    torch.manual_seed(1)

    vocab_size = 8
    model = LandauerLanguageModel(vocab_size, embed_dim=5, hidden_dim=7, num_layers=2)

    idx = torch.randint(0, vocab_size, (2, 4))
    logits, hidden_trace, h_stack = model(idx)

    assert logits.shape == (2, 4, vocab_size)
    assert len(hidden_trace) == 4
    assert h_stack.shape == (2, 2, 7)

    generated = model.generate(idx.clone(), max_new_tokens=3, h=h_stack)
    assert generated.shape == (2, 7)
    assert torch.equal(generated[:, :4], idx)
