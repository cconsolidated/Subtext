import unittest

from subtext.models import choose_device, choose_dtype


class Availability:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeTorch:
    def __init__(self, cuda=False, mps=False):
        self.cuda = Availability(cuda)
        self.backends = type("Backends", (), {"mps": Availability(mps)})()
        self.bfloat16 = "bf16"
        self.float16 = "fp16"
        self.float32 = "fp32"


class ModelLoadingTests(unittest.TestCase):
    def test_device_selection_prefers_cuda_then_mps(self):
        self.assertEqual(choose_device(FakeTorch(cuda=True, mps=True), "auto"), "cuda")
        self.assertEqual(choose_device(FakeTorch(mps=True), "auto"), "mps")
        self.assertEqual(choose_device(FakeTorch(), "auto"), "cpu")

    def test_auto_dtype_matches_historical_behavior(self):
        torch = FakeTorch()
        self.assertEqual(choose_dtype(torch, "auto", "cpu"), "fp32")
        self.assertEqual(choose_dtype(torch, "auto", "mps"), "bf16")


if __name__ == "__main__":
    unittest.main()
