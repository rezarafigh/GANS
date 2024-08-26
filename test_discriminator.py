import unittest
import torch
from src.discriminator import Discriminator

class TestDiscriminator(unittest.TestCase):
    def test_output_shape(self):
        """Test that the discriminator outputs the correct shape given an image batch."""
        images = torch.randn(10, 1, 28, 28)  # Batch size of 10, 1 channel, 28x28 images
        discriminator = Discriminator()
        outputs = discriminator(images)
        self.assertEqual(outputs.size(), (10, 1))

    def test_output_values(self):
        """Test that the discriminator outputs a probability between 0 and 1."""
        images = torch.randn(10, 1, 28, 28)
        discriminator = Discriminator()
        outputs = discriminator(images)
        self.assertTrue(torch.all(outputs >= 0) and torch.all(outputs <= 1))

if __name__ == '__main__':
    unittest.main()
