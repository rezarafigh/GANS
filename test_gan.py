import unittest
import torch
from src.generator import Generator
from src.discriminator import Discriminator

class TestGAN(unittest.TestCase):
    def test_gan_integration(self):
        """Test the integration of generator and discriminator in the GAN setup."""
        generator = Generator()
        discriminator = Discriminator()
        noise = torch.randn(10, 100)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        self.assertEqual(outputs.size(), (10, 1))
        self.assertTrue(torch.all(outputs >= 0) and torch.all(outputs <= 1))

if __name__ == '__main__':
    unittest.main()
