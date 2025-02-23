import base64
import orjson
import torch

from safetensors.torch import _tobytes

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
}


class TensorBinary:
    @staticmethod
    def deserialize(body, query):
        if "dtype" not in query:
            raise ValueError("Expected `dtype` in query params.")
        dtype = query["dtype"]
        if dtype not in DTYPE_MAP:
            raise ValueError(
                f"Supported `dtype`: {list(DTYPE_MAP.keys())}. Got `{dtype}`."
            )
        torch_dtype = DTYPE_MAP[dtype]
        if "shape" not in query:
            raise ValueError("Expected `shape` in query params.")
        shape = query["shape"]
        tensor = torch.frombuffer(bytearray(body), dtype=torch_dtype).reshape(shape)
        return {"inputs": tensor}

    @staticmethod
    def serialize(tensor, accept=None):
        if isinstance(tensor, torch.Tensor):
            tensor_data = _tobytes(tensor, "inputs")
            shape = orjson.dumps(list(tensor.shape)).decode("utf-8")
            dtype = str(tensor.dtype).split(".")[-1]
            headers = {"shape": shape, "dtype": dtype}
            return tensor_data, headers
        else:
            raise ValueError(f"Can only serialize torch.Tensor, got {type(tensor)}")


class TensorBase64:
    @staticmethod
    def deserialize(body, query):
        data = orjson.loads(body)
        if "inputs" not in data:
            raise ValueError("Expected `inputs` in data.")
        inputs = base64.b64decode(data["inputs"])
        if "dtype" not in query:
            raise ValueError("Expected `dtype` in query params.")
        dtype = query["dtype"]
        if dtype not in DTYPE_MAP:
            raise ValueError(
                f"Supported `dtype`: {list(DTYPE_MAP.keys())}. Got `{dtype}`."
            )
        torch_dtype = DTYPE_MAP[dtype]
        if "shape" not in query:
            raise ValueError("Expected `shape` in query params.")
        shape = query["shape"]
        tensor = torch.frombuffer(bytearray(inputs), dtype=torch_dtype).reshape(shape)
        return {"inputs": tensor}

    @staticmethod
    def serialize(tensor, accept=None):
        if isinstance(tensor, torch.Tensor):
            tensor_data = base64.b64encode(_tobytes(tensor, "inputs")).decode("utf-8")
            shape = list(tensor.shape)
            dtype = str(tensor.dtype).split(".")[-1]
            data = orjson.dumps(
                {"inputs": tensor_data, "parameters": {"shape": shape, "dtype": dtype}}
            )
            return data, {}
        else:
            raise ValueError(f"Can only serialize torch.Tensor, got {type(tensor)}")
