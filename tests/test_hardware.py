import pytest

from pycoder.hardware import Hardware, detect_hardware, suggest_profile


@pytest.mark.parametrize(
    'total,free,bf16,batch,precision',
    [(8, 8, True, 1, 'bfloat16'), (12, 12, True, 2, 'bfloat16'),
     (24, 24, True, 4, 'bfloat16'), (32, 32, True, 8, 'bfloat16'),
     (24, 6, True, 1, 'bfloat16'), (8, 8, False, 1, 'float16')],
)
def test_cuda_profile_budget(total, free, bf16, batch, precision):
    config, _ = suggest_profile(Hardware('cuda', 'test', 32, total, free, bf16))
    assert config.batch_size == batch
    assert config.precision == precision
    assert config.batch_size * config.gradient_accumulation * 1024 == 65536


def test_low_memory_warning():
    _, notes = suggest_profile(Hardware('cuda', 'test', 4, 8, 2, True))
    assert any('Low free VRAM' in note for note in notes)
    assert any('Limited RAM' in note for note in notes)


@pytest.mark.parametrize('device', ['mps', 'cpu'])
def test_non_cuda_profile(device):
    config, _ = suggest_profile(Hardware(device, 'test', 16))
    assert config.precision == 'float32'
    assert config.batch_size == 1


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    monkeypatch.setattr('torch.backends.mps.is_available', lambda: False)
    assert detect_hardware().device == 'cpu'
    with pytest.raises(ValueError, match='CUDA'):
        detect_hardware('cuda')
    with pytest.raises(ValueError, match='MPS'):
        detect_hardware('mps')


def test_cuda_detection(monkeypatch):
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('torch.cuda.current_device', lambda: 0)
    monkeypatch.setattr('torch.cuda.mem_get_info', lambda index: (6 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr('torch.cuda.get_device_name', lambda index: 'test NVIDIA')
    monkeypatch.setattr('torch.cuda.is_bf16_supported', lambda: True)
    hardware = detect_hardware()
    assert hardware.name == 'test NVIDIA'
    assert hardware.vram_gib == 8
    assert hardware.free_vram_gib == 6
    assert hardware.bf16


def test_cli_saves_valid_config_without_overwriting(monkeypatch, tmp_path):
    from pycoder.config import TrainConfig, load_config
    from pycoder.hardware import main

    output = tmp_path / 'profile.yaml'
    monkeypatch.setattr('pycoder.hardware.detect_hardware',
                        lambda device: Hardware('cuda', 'test', 32, 8, 8, True))
    monkeypatch.setattr('sys.argv', ['hardware', '--output', str(output)])
    main()
    config = load_config(output, TrainConfig)
    assert config.precision == 'bfloat16'
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_bytes() == original
