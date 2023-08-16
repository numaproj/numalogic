import io
import pickle
from typing import Union

import torch

from numalogic.tools.types import artifact_t, state_dict_t


def dumps(
    deserialized_object: Union[artifact_t, state_dict_t],
    pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
) -> bytes:
    buffer = io.BytesIO()
    torch.save(deserialized_object, buffer, pickle_protocol=pickle_protocol)
    serialized_obj = buffer.getvalue()
    buffer.close()
    return serialized_obj


def loads(serialized_object: bytes) -> Union[artifact_t, state_dict_t]:
    buffer = io.BytesIO(serialized_object)
    deserialized_obj = torch.load(buffer)
    buffer.close()
    return deserialized_obj
