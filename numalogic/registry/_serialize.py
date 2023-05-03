import io
import torch


def dumps(deserialized_object):
    buf = io.BytesIO()
    torch.save(deserialized_object, buf)
    return buf.getvalue()


def loads(serialized_object):
    buffer = io.BytesIO(serialized_object)
    return torch.load(buffer)
