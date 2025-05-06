import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Register pytest markers
def pytest_configure(config):
    config.addinivalue_line("markers", "order: mark test execution order") 