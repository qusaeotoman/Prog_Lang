from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, TextIO
import sys
from pathlib import Path
import io


def field_name_of(node):
    p = node.parent
    if p is None:
        return node.type
    for i in range(p.child_count):
        if p.child(i) == node and p.field_name_for_child(i):
            return p.field_name_for_child(i)
    return node.type


@dataclass
class TreeViewNode:
    """
    Узел логического дерева, соответствующий тому, что раньше выводилось в файл.

    label  — то, что раньше писалось в строку (identifier, "(", "hello", list<argDef> (empty), ...).
    node   — исходный node из парсера (для "настоящих" узлов); для синтетических узлов (empty) — None.
    children — потомки в отображаемом дереве.
    type   — тип данных узла (заполняется при типизации).
    """

    label: str
    node: Any | None
    children: List["TreeViewNode"]
    type: str | None = None


def build_tree_view(root) -> tuple[TreeViewNode, list[str]]:
    """
    Принимает корневой node парсера и возвращает кортеж:
    (дерево TreeViewNode, список строк с описанием ошибок).
    """

    replace_names = {
        "list_argDef": "list<argDef>",
        "list_identifier": "list<identifier>",
        "statment_block": "statment.block",
        "listExpr": "list<expr>",
    }

    # типы, которые считались "токенами" в исходной функции
    TOKEN_TYPES = {
        "identifier",
        "bin_op",
        "str",
        "char",
        "hex",
        "bits",
        "dec",
        "un_op",
        "builtin",
    }
    
    errors: list[str] = []
    
    def _child_anc(anc_has_next: list[bool], is_last: bool) -> list[bool]:
        return [*anc_has_next, not is_last]

    def _build(n, anc_has_next: list[bool], is_last: bool) -> TreeViewNode:
        fname = field_name_of(n)
        if fname in replace_names:
            fname = replace_names[fname]

        is_token = n.type in TOKEN_TYPES

        if getattr(n, "is_missing", False):
            errors.append(
                f'Error: missing element "{n.type}" in end point {n.end_point}'
            )
        if getattr(n, "is_error", False):
            errors.append(f"Error: incorrect in end point {n.end_point}")

        if len(n.children) == 0:
            text = (
                n.text.decode("utf-8")
                if isinstance(n.text, (bytes, bytearray))
                else str(n.text)
            )

            if is_token:
                parent = TreeViewNode(label=fname, node=n, children=[])
                parent.children.append(
                    TreeViewNode(label=f'"{text}"', node=n, children=[])
                )
                return parent

            if not anc_has_next:
                label = fname
            else:
                label = f'"{text}"'

            return TreeViewNode(label=label, node=n, children=[])

        label = fname
        children_view: list[TreeViewNode] = []

        child_anc = _child_anc(anc_has_next, is_last)

        inject_cfg = {
            "funcSignature": ("list_argDef", "("),
            "call_expression": ("listExpr", "("),
            "indexer": ("listExpr", "["),
        }

        need_field, left_token = inject_cfg.get(n.type, (None, None))
        present_fields = {n.field_name_for_child(i) for i in range(n.child_count)} - {
            None
        }
        should_inject = bool(need_field) and (need_field not in present_fields)
        injected = False

        for i, ch in enumerate(n.children):
            ch_is_last = i == len(n.children) - 1
            child_view = _build(ch, child_anc, ch_is_last)
            children_view.append(child_view)

            if should_inject and not injected and ch.type == left_token:
                display = replace_names.get(need_field, need_field)
                children_view.append(
                    TreeViewNode(label=f"{display}", node=None, children=[])
                )
                injected = True

        return TreeViewNode(label=label, node=n, children=children_view)

    return _build(root, [], True), errors


def print_tree_view(
    root: TreeViewNode,
    *,
    ascii: bool = False,
    out: TextIO = sys.stdout,
) -> None:
    """
    Рисует дерево TreeViewNode в поток `out`.
    По умолчанию выводит в консоль (sys.stdout).
    Можно передать файл или любой другой текстовый поток.
    """
    VBAR, SPACE, TEE, ELBOW = (
        ("|   ", "    ", "|-- ", "`-- ") if ascii else ("│   ", "    ", "├── ", "└── ")
    )

    def _prefix(anc_has_next: list[bool], is_last: bool) -> str:
        if not anc_has_next:
            return ""
        base = "".join(VBAR if has_next else SPACE for has_next in anc_has_next)
        return base + (ELBOW if is_last else TEE)

    def _child_anc(anc_has_next: list[bool], is_last: bool) -> list[bool]:
        return [*anc_has_next, not is_last]

    def _walk(node: TreeViewNode, anc_has_next: list[bool], is_last: bool) -> None:
        out.write(_prefix(anc_has_next, is_last) + node.label + "\n")
        child_anc = _child_anc(anc_has_next, is_last)
        for i, ch in enumerate(node.children):
            _walk(ch, child_anc, i == len(node.children) - 1)

    _walk(root, [], True)


def write_tree_view_to_file(
    root: TreeViewNode,
    out_path: str | Path,
    *,
    ascii: bool = False,
) -> None:
    """
    Удобная обёртка: принимает дерево и путь к файлу,
    создаёт директории (если нужно) и пишет туда дерево.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        print_tree_view(root, ascii=ascii, out=f)


def tree_view_to_str(
    root: TreeViewNode,
    *,
    ascii: bool = False,
) -> str:
    """
    Возвращает строковое представление дерева TreeViewNode
    в том же формате, что и print_tree_view.
    """
    buf = io.StringIO()
    print_tree_view(root, ascii=ascii, out=buf)
    return buf.getvalue()
