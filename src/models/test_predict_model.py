import unittest
import os
import torch
from unittest.mock import patch, MagicMock
from simple_cnn import SimpleCNN
import predict_model  # Import the script you want to test

class TestPredictModel(unittest.TestCase):
    def setUp(self):
        self.image_path = 'sample_image.jpg'
        self.model_path = '../../models/cifar10_simple_cnn.pth'

        # Set up a dummy environment
        os.environ["YOUR_WORKSPACE/YOUR_PROJECT_NAME"] = "dummy_workspace/dummy_project"
        os.environ["YOUR_API_TOKEN"] = "dummy_token"

        # Mock Neptune to prevent actual network calls
        self.neptune_mock = MagicMock()
        predict_model.neptune = self.neptune_mock

        # Mock the model loading to prevent file not found error
        dummy_model = SimpleCNN()
        torch.save(dummy_model.state_dict(), self.model_path)
        self.addCleanup(os.remove, self.model_path)  # Ensure the file is deleted after the tests

    def test_env_variable_loading(self):
        self.assertEqual(predict_model.NEPTUNE_WORKSPACE_PROJECT, "dummy_workspace/dummy_project")
        self.assertEqual(predict_model.NEPTUNE_API_TOKEN, "dummy_token")

    def test_neptune_initialization(self):
        self.neptune_mock.init.assert_called_once_with("dummy_workspace/dummy_project", api_token="dummy_token")

    def test_model_loading(self):
        self.assertIsInstance(predict_model.model, SimpleCNN)
        self.assertTrue(predict_model.model.training == False)  # Model should be in evaluation mode

    @patch('predict_model.Image.open')  # Mock the Image.open call to prevent file not found error
    def test_prediction(self, mock_open):
        dummy_image = torch.randn(1, 3, 32, 32)
        mock_open.return_value = dummy_image
        prediction, confidence = predict_model.predict(self.image_path)
        self.assertIsInstance(prediction, int)
        self.assertIsInstance(confidence, float)

    @patch('argparse.ArgumentParser.parse_args')
    def test_command_line_argument_parsing(self, mock_args):
        mock_args.return_value = MagicMock(image_path=self.image_path)
        with self.assertRaises(SystemExit):  # Prevents the script from exiting
            predict_model.main()

if __name__ == '__main__':
    unittest.main()
