import base64
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from pydantic_core import core_schema
from typing_extensions import Annotated

from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ValidationError,
)
from pydantic.json_schema import JsonSchemaValue


class NumpyArrayModel(BaseModel):
    data: str
    dtype: str
    shape: Tuple[int, ...]

    @classmethod
    def from_ndarray(cls, obj: np.ndarray):
        data = base64.b64encode(np.ascontiguousarray(obj).data).decode('ascii')
        return cls(data=data, dtype=obj.dtype.str, shape=obj.shape)


class _NumpyPydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        We return a pydantic_core.CoreSchema that behaves in the following ways:

        * `NumpyArrayModel` (and JSON dumps of them) instances will be parsed into `np.ndarray` instances
        * `np.ndarray` instances will be parsed as `np.ndarray` instances without any changes
        * Nothing else will pass validation
        * Serialization will always return something with the schema of `NumpyArrayModel`
        """

        def validate_from_array_model(value: NumpyArrayModel | dict) -> np.ndarray:
            if isinstance(value, dict):
                value = NumpyArrayModel(**value)
            decoded_data = base64.b64decode(value.data)
            result = np.frombuffer(decoded_data, dtype=np.dtype(value.dtype)).reshape(value.shape)
            return result

        from_array_model_schema = core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(validate_from_array_model),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_array_model_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(np.ndarray),
                    from_array_model_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                NumpyArrayModel.from_ndarray
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Use the same schema that would be used for `NumpyArrayModel`
        return handler(NumpyArrayModel.schema())

# We now create an `Annotated` wrapper that we'll use as the annotation for fields on `BaseModel`s, etc.
PydanticNumpyArray = Annotated[
    np.ndarray, _NumpyPydanticAnnotation
]


# Create a model class that uses this annotation as a field
class Model(BaseModel):
    array: PydanticNumpyArray


def test():
    array = np.random.rand(3,4)
    m = Model(array = array)
    assert m.array is array
    array_model = NumpyArrayModel.from_ndarray(array)
    assert m.model_dump()['array'] == array_model.model_dump()
    
    dumped_model = m.model_dump()
    json_dumped_model = m.model_dump_json()
    assert Model.model_validate(dumped_model).model_dump_json() == json_dumped_model
    assert np.allclose(Model.model_validate_json(json_dumped_model).array, array)
    
if __name__ == '__main__':
    test()