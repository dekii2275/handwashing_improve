import numpy as np
import torch

from scripts.run_paper_cnn import DeviceWindows, PaperCNN, Windows


def test_windows_filter_original_missing_frames_and_use_final_label():
    x = np.arange(7 * 126, dtype=np.float32).reshape(7, 126)
    presence = np.ones((7, 2), dtype=np.int32)
    presence[2] = 0
    labels = np.arange(7, dtype=np.int64)
    windows = Windows([x], [presence], [labels], window=3)
    assert len(windows) == 2
    first, label = windows[0]
    np.testing.assert_array_equal(first.numpy(), x[3:6])
    assert label == 5
    assert windows[1][1] == 6
    device_windows = DeviceWindows(windows, torch.device("cpu"))
    batch, targets = next(device_windows.batches())
    np.testing.assert_array_equal(batch[0].numpy(), x[3:6])
    np.testing.assert_array_equal(targets.numpy(), [5, 6])


def test_paper_cnn_no_batch_norm_and_correct_shape():
    model = PaperCNN()
    assert not any(isinstance(layer, torch.nn.BatchNorm1d) for layer in model.modules())
    assert model(torch.randn(4, 48, 126)).shape == (4, 7)
