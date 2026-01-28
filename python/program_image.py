from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

@dataclass
class Instruction:
    mnemonic: str
    operands: List[str] = field(default_factory=list)
    comment: Optional[str] = None

    def to_text(self) -> str:
        s = self.mnemonic
        if self.operands:
            s += " " + ", ".join(self.operands)
        if self.comment:
            s += f"      ; {self.comment}"
        return s

@dataclass
class DataItem:
    name: str
    literal: Optional[Union[int, str, bytes]] = None
    size_bytes: Optional[int] = None

@dataclass
class CodeFragment:
    name: str                   # label name
    instructions: List[Instruction] = field(default_factory=list)

@dataclass
class ProgramImage:
    data_items: Dict[str, DataItem] = field(default_factory=dict)
    code_fragments: Dict[str, CodeFragment] = field(default_factory=dict)

    def add_fragment(self, frag: CodeFragment) -> None:
        self.code_fragments[frag.name] = frag

    def add_data(self, item: DataItem) -> None:
        self.data_items[item.name] = item
