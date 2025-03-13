"""
Tests for utility functions.

This module contains tests for the utility functions in the Video Timeline Analyzer.
"""

import unittest
import torch
from unittest.mock import patch, MagicMock

from src.utils.gpu_utils import detect_gpu, get_optimal_device


class TestGPUUtils(unittest.TestCase):
    """Tests for GPU utility functions."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_capability')
    @patch('torch.cuda.get_device_properties')
    def test_detect_gpu_with_cuda(self, mock_properties, mock_capability, 
                                 mock_name, mock_count, mock_available):
        """Test GPU detection with CUDA available."""
        # Mock CUDA availability
        mock_available.return_value = True
        mock_count.return_value = 1
        mock_name.return_value = "Test GPU"
        mock_capability.return_value = (7, 5)
        
        # Mock device properties
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1e9  # 8GB
        mock_properties.return_value = mock_props
        
        has_gpu, gpu_type, device_count = detect_gpu()
        
        self.assertTrue(has_gpu)
        self.assertEqual(gpu_type, 'cuda')
        self.assertEqual(device_count, 1)
    
    @patch('torch.cuda.is_available')
    @patch('platform.system')
    def test_detect_gpu_no_gpu(self, mock_system, mock_available):
        """Test GPU detection with no GPU available."""
        # Mock no CUDA availability
        mock_available.return_value = False
        mock_system.return_value = 'Linux'
        
        has_gpu, gpu_type, device_count = detect_gpu()
        
        self.assertFalse(has_gpu)
        self.assertIsNone(gpu_type)
        self.assertEqual(device_count, 0)
    
    @patch('src.utils.gpu_utils.detect_gpu')
    def test_get_optimal_device_cuda(self, mock_detect_gpu):
        """Test getting optimal device with CUDA available."""
        mock_detect_gpu.return_value = (True, 'cuda', 1)
        
        device = get_optimal_device()
        
        self.assertEqual(device.type, 'cuda')
    
    @patch('src.utils.gpu_utils.detect_gpu')
    def test_get_optimal_device_cpu(self, mock_detect_gpu):
        """Test getting optimal device with no GPU available."""
        mock_detect_gpu.return_value = (False, None, 0)
        
        device = get_optimal_device()
        
        self.assertEqual(device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()