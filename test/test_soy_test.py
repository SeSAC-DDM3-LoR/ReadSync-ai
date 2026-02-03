import unittest
import os
import sys
# Mock aioboto3 before import
from unittest.mock import MagicMock, AsyncMock
sys.modules["aioboto3"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from app.domain.soy_test.services.file_service import LocalFileService
from app.domain.soy_test.services.soy_test_service import SoyTestService
from app.domain.soy_test.services.backend_integration_service import BackendIntegrationService

class TestSoyTestService(unittest.IsolatedAsyncioTestCase):
    async def test_local_file_service_read(self):
        # Create a dummy file
        test_dir = os.path.join(os.path.dirname(__file__), "resources")
        os.makedirs(test_dir, exist_ok=True)
        test_file_path = os.path.join(test_dir, "test_file.json")
        with open(test_file_path, "w") as f:
            f.write('{"key": "value"}')

        service = LocalFileService(base_directory=test_dir)
        content = await service.read_file("test_file")
        self.assertEqual(content, {"key": "value"})
        
        # Cleanup
        os.remove(test_file_path)
        os.rmdir(test_dir)

    async def test_soy_test_service_process(self):
        # Mock FileService
        mock_file_service = MagicMock()
        mock_file_service.read_file = AsyncMock(return_value={"data": "test"})
        
        # Mock BackendIntegrationService
        mock_backend_service = MagicMock()
        mock_backend_service.send_data = AsyncMock(return_value={"status": "received"})

        service = SoyTestService(mock_file_service, mock_backend_service)
        result = await service.process_file("dummy")
        
        self.assertEqual(result["file_content"], {"data": "test"})
        self.assertEqual(result["backend_response"], {"status": "received"})
        
        mock_file_service.read_file.assert_called_once_with("dummy")
        mock_backend_service.send_data.assert_called_once()

if __name__ == "__main__":
    unittest.main()
