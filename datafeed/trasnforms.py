from dataclasses import dataclass
from abc import ABC, abstractmethod
from schema import Schema
from typing import Dict, Any, Type, List, Sequence
import numpy as np

REGISTERED_TRANSFORM_CLASSES = {}


class Transform(ABC):
    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    @abstractmethod
    def input_schema(self) -> Schema:
        raise NotImplementedError

    def dry_run(self, items: Dict[str, Any]):
        self.input_schema.validate(items)
        return self._dry_run(items)

    def __call__(self, items: Dict[str, Any]):
        self.input_schema.validate(items)
        return self.forward(items)

    @abstractmethod
    def _dry_run(self, items: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def forward(self, items: Dict[str, Any]):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


def register_transform(cls: Type[Transform]):
    global REGISTERED_TRANSFORM_CLASSES
    name = cls.__name__
    assert name not in REGISTERED_TRANSFORM_CLASSES, f"exists class: {REGISTERED_TRANSFORM_CLASSES}"
    REGISTERED_TRANSFORM_CLASSES[name] = cls
    return cls


def get_transform(name: str) -> Type[Transform]:
    global REGISTERED_TRANSFORM_CLASSES
    assert name in REGISTERED_TRANSFORM_CLASSES, f"available class: {REGISTERED_TRANSFORM_CLASSES}"
    return REGISTERED_TRANSFORM_CLASSES[name]


def is_valid_points(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 3


@dataclass
@register_transform
class Compose(Transform):
    transforms: List[Transform]

    def forward(self, items: Dict[str, Any]):
        for t in self.transforms:
            items = t(items)
            if items is None:
                print(f"Transform {t.name} returned None")
                return None
        return items

    def _dry_run(self, items: Dict[str, Any]):
        for t in self.transforms:
            items = t.dry_run(items)
            if items is None:
                return None
        return items

    @property
    def input_schema(self) -> Schema:
        if len(self.transforms) > 0:
            return self.transforms[0].input_schema
        else:
            return Schema({}, ignore_extra_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        transforms = []
        for name, kwargs in d.items():
            transforms.append(get_transform(name).from_dict(kwargs))
        # noinspection PyArgumentList
        return cls(transforms)

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for t in self.transforms:
            d[t.name] = t.to_dict()
        return d


@dataclass
class NumpyTransform(Transform, ABC):
    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        kwargs = {}
        for name, param in d.items():
            if isinstance(param, Sequence):
                kwargs[name] = np.array(param)
            else:
                kwargs[name] = param
        # noinspection PyArgumentList
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for name, param in self.__dict__.items():
            if isinstance(param, np.ndarray) or isinstance(param, np.number):
                d[name] = param.tolist()
            else:
                d[name] = param
        return d
