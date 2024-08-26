import unittest
import torch
from src.generator import Generator

class TestGenerator(unittest.TestCase):
    def test_output_shape(self):
        """Test that the generator outputs the correct shape based on the input noise."""
        noise = torch.randn(10, 100)  # Batch size of 10, noise vector size 100
        generator = Generator()
        generated_images = generator(noise)
        self.assertEqual(generated_images.size(), (10, 1, 28, 28))

    def test_output_range(self):
        """Test that the generator outputs values in the expected range of -1 to 1."""
        noise = torch.randn(10, 100)
        generator = Generator()
        generated_images = generator(noise)
        self.assertTrue(torch.max(generated_images) <= 1)
        self.assertTrue(torch.min(generated_images) >= -1)

if __name__ == '__main__':
    unittest.main()
