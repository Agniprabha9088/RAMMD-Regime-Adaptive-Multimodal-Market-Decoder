"""
Unit tests for RAMMD models
"""

import pytest
import torch
import sys
sys.path.append('src')

from models.rammd import RAMMD
from models.regime_detection import RegimeDetector
from models.encoders import PriceEncoder, NewsEncoder, SocialEncoder, MacroEncoder


def test_regime_detector():
    """Test regime detection module"""
    detector = RegimeDetector(n_regimes=4)
    
    # Test feature extraction
    returns = torch.randn(10, 100)
    features = detector.extract_regime_features(returns)
    
    assert features.shape[1] > 0
    assert not torch.isnan(features).any()


def test_price_encoder():
    """Test price encoder"""
    encoder = PriceEncoder(window_size=252, input_channels=23, d_model=512)
    
    x = torch.randn(4, 252, 23)
    output = encoder(x)
    
    assert output.shape == (4, 512)
    assert not torch.isnan(output).any()


def test_rammd_forward():
    """Test complete RAMMD forward pass"""
    config = {
        'model': {
            'num_regimes': 4,
            'device': 'cpu',
            'encoders': {},
            'moe': {},
            'contrastive': {},
            'gnn': {},
            'wavelet': {},
            'fusion': {},
            'prediction': {}
        }
    }
    
    model = RAMMD(config['model'])
    model.eval()
    
    # Create dummy input
    price = torch.randn(2, 252, 23)
    macro = torch.randn(2, 24)
    
    with torch.no_grad():
        outputs = model(price_data=price, macro_data=macro)
    
    assert 'regression_output' in outputs
    assert outputs['regression_output'].shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__])
