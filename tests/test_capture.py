import unittest

from subtext.capture import ActivationCapture, CaptureError


class FakeTensor:
    def __init__(self, data, calls=()):
        self.data = data
        self.calls = calls

    def __getitem__(self, key):
        value = self.data
        keys = key if isinstance(key, tuple) else (key,)
        for item in keys:
            value = value[item]
        return FakeTensor(value, self.calls)

    def detach(self):
        return FakeTensor(self.data, self.calls + ("detach",))

    def clone(self):
        return FakeTensor(self.data, self.calls + ("clone",))

    def cpu(self):
        return FakeTensor(self.data, self.calls + ("cpu",))


class FakeHandle:
    def __init__(self, layer):
        self.layer = layer

    def remove(self):
        self.layer.hook = None


class FakeLayer:
    def __init__(self):
        self.hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return FakeHandle(self)

    def fire(self, output):
        if self.hook is not None:
            self.hook(self, (), output)


class FakeAdapter:
    def __init__(self):
        self.layers = [FakeLayer(), FakeLayer()]

    def get_hook_module(self, layer, hook_point):
        if hook_point != "block_output":
            raise KeyError(hook_point)
        return self.layers[layer]

    def forward(self, input_ids):
        for layer_number, layer in enumerate(self.layers):
            layer.fire(FakeTensor([[[layer_number, 0], [layer_number, 1]]]))
        return {"logits": input_ids}


class CaptureTests(unittest.TestCase):
    def test_captures_named_positions_and_moves_them_to_cpu(self):
        adapter = FakeAdapter()
        baseline = adapter.forward([10, 11])
        result = ActivationCapture(adapter, [0, 1], "block_output").run(
            [10, 11], {"end": 1}
        )
        self.assertEqual(result.output, baseline)
        self.assertEqual(result.activations["end"][1].data, [1, 1])
        self.assertEqual(
            result.activations["end"][1].calls, ("detach", "clone", "cpu")
        )
        self.assertTrue(all(layer.hook is None for layer in adapter.layers))

    def test_rejects_non_integer_position(self):
        with self.assertRaises(CaptureError):
            ActivationCapture(FakeAdapter(), [0], "block_output").run(
                [10], {"end": "last"}
            )

    def test_rejects_negative_position(self):
        with self.assertRaises(CaptureError):
            ActivationCapture(FakeAdapter(), [0], "block_output").run(
                [10], {"end": -1}
            )


if __name__ == "__main__":
    unittest.main()
