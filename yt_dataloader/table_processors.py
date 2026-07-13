from abc import ABC, abstractmethod


class TableProcessor(ABC):

    def __init__(self,) -> None:
        super().__init__()


    @abstractmethod
    def decode_fn(self, row: dict[bytes, bytes]):
        ...

    @staticmethod
    @abstractmethod
    def collate_fn(batch):
        ...
