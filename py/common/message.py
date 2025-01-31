# Adapted from src/message/message.{hh,cc}

import enum
import struct

from typing import Union, Tuple


class Message:
    HEADER_LENGTH = 5

    class OpCode(enum.Enum):
        Hey = 0x1
        Ping = enum.auto()
        Bye = enum.auto()

        InitializeWorker = enum.auto()
        AckInitialize = enum.auto()
        SetRoute = enum.auto()
        AckRoute = enum.auto()
        PushDummyPrompts = enum.auto()
        PushPrompts = enum.auto()
        PushCompletions = enum.auto()
        BatchedInferenceState = enum.auto()

    def __init__(self, opcode: Union[OpCode, int], payload: bytes):
        if isinstance(opcode, int):
            self.opcode = Message.OpCode(opcode)
        elif isinstance(opcode, Message.OpCode):
            self.opcode = opcode
        else:
            raise ValueError("Invalid opcode")

        self.payload = payload

    def serialize_header(self):
        pass

    @staticmethod
    def parse_header(header: bytes) -> Tuple[int, int]:
        if len(header) != Message.HEADER_LENGTH:
            raise ValueError("Invalid header length")

        payload_length, opcode = struct.unpack("!IB", header)
        return (payload_length, opcode)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Message(opcode={self.opcode}, payload.len={len(self.payload)})"

    def serialize(self) -> bytes:
        header = struct.pack("!IB", len(self.payload), self.opcode.value)
        return header + self.payload
