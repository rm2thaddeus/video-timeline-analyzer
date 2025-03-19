"""
📌 Purpose: Test suite for GPU utilities module
🔄 Latest Changes: Initial implementation of GPU utility tests
⚙️ Key Logic: Tests CUDA setup, memory management, and device operations
📂 Expected File Path: tests/test_gpu_utils.py
🧠 Reasoning: Ensures reliability of GPU-related operations
"""

import pytest
import torch
import torch.nn as nn
from src.utils.gpu_utils import (
    setup_cuda,
    clear_gpu_memory,
    get_gpu_memory_info,
    optimize_gpu_memory,
    batch_to_device,
    get_optimal_batch_size
)

# Skip tests if CUDA is not available
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available"
)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.mean([2, 3])
        return self.fc(x)

@requires_cuda
def test_setup_cuda():
    device = setup_cuda(gpu_memory_fraction=0.5)
    assert device.type == "cuda"
    assert torch.cuda.is_available()
    assert torch.backends.cudnn.benchmark

def test_setup_cuda_cpu_fallback(monkeypatch):
    # Mock torch.cuda.is_available to return False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = setup_cuda()
    assert device.type == "cpu"

@requires_cuda
def test_clear_gpu_memory():
    # Allocate some tensors
    tensors = [torch.randn(1000, 1000).cuda() for _ in range(5)]
    initial_memory = torch.cuda.memory_allocated()
    assert initial_memory > 0
    
    # Clear memory
    clear_gpu_memory()
    del tensors
    
    # Check memory was cleared
    final_memory = torch.cuda.memory_allocated()
    assert final_memory < initial_memory

@requires_cuda
def test_get_gpu_memory_info():
    memory_info = get_gpu_memory_info()
    assert isinstance(memory_info, dict)
    assert all(key in memory_info for key in ['total_gb', 'reserved_gb', 'allocated_gb', 'free_gb'])
    assert all(isinstance(value, float) for value in memory_info.values())
    assert memory_info['total_gb'] > 0

def test_get_gpu_memory_info_cpu():
    if not torch.cuda.is_available():
        memory_info = get_gpu_memory_info()
        assert memory_info is None

@requires_cuda
def test_optimize_gpu_memory():
    model = SimpleModel()
    optimized_model = optimize_gpu_memory(model, use_half_precision=True)
    
    assert next(optimized_model.parameters()).dtype == torch.float16
    assert next(optimized_model.parameters()).device.type == "cuda"

@requires_cuda
def test_batch_to_device():
    # Test tensor
    tensor = torch.randn(2, 3)
    device = torch.device("cuda")
    tensor_gpu = batch_to_device(tensor, device)
    assert tensor_gpu.device.type == "cuda"
    
    # Test list
    tensor_list = [torch.randn(2, 3) for _ in range(3)]
    list_gpu = batch_to_device(tensor_list, device)
    assert all(t.device.type == "cuda" for t in list_gpu)
    
    # Test dict
    tensor_dict = {"a": torch.randn(2, 3), "b": torch.randn(3, 4)}
    dict_gpu = batch_to_device(tensor_dict, device)
    assert all(t.device.type == "cuda" for t in dict_gpu.values())

@requires_cuda
def test_get_optimal_batch_size():
    model = SimpleModel().cuda()
    input_shape = (3, 32, 32)
    batch_size = get_optimal_batch_size(model, input_shape, target_memory_usage=0.7)
    
    assert isinstance(batch_size, int)
    assert batch_size > 0
    
    # Test with batch size
    try:
        sample_batch = torch.randn(batch_size, *input_shape).cuda()
        with torch.no_grad():
            output = model(sample_batch)
        assert output.shape[0] == batch_size
    except RuntimeError:
        pytest.fail("Failed to process optimal batch size") 