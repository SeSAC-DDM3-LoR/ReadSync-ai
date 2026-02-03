import unittest
import os
import sys
import shutil
# Mock aioboto3 before import
from unittest.mock import MagicMock, AsyncMock
sys.modules["aioboto3"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from app.domain.soy_test.services.file_service import LocalFileService

class TestFileServiceRecursive(unittest.IsolatedAsyncioTestCase):
    async def test_recursive_search(self):
        # Create a test directory structure
        test_dir = os.path.join(os.path.dirname(__file__), "recursive_resources")
        os.makedirs(os.path.join(test_dir, "subdir"), exist_ok=True)
        
        test_file_path = os.path.join(test_dir, "subdir", "deep_file.json")
        with open(test_file_path, "w") as f:
            f.write('{"key": "deep_value"}')

        service = LocalFileService(base_directory=test_dir)
        
        # Test finding without subdir path
        content = await service.read_file("deep_file")
        self.assertEqual(content, {"key": "deep_value"})
        
        # Test finding with extension
        content = await service.read_file("deep_file.json")
        self.assertEqual(content, {"key": "deep_value"})

        # Cleanup
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    unittest.main()
