from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, TextIO
from tree_parser import TreeViewNode

def strip_quotes(label: str) -> str:
    """Убираем кавычки у лексем вида '"x"'."""
    if len(label) >= 2 and label[0] == label[-1] == '"':
        return label[1:-1]
    return label

def extract_place_name_from_expr(expr_node: TreeViewNode, where: str) -> str:
    """
    expr_node: узел с label == 'expr', внутри которого ожидаем place.
    where: текст для сообщения об ошибке.
    """
    if expr_node.label != 'expr':
        raise ValueError(f'{where}: ожидается expr, а не {expr_node.label!r}')

    if not expr_node.children:
        raise ValueError(f'{where}: expr без детей')

    inner = expr_node.children[0]

    if inner.label != 'place':
        raise ValueError(
            f'{where}: ожидается place (простая переменная), '
            f'а не {inner.label!r}'
        )

    if not inner.children:
        raise ValueError(f'{where}: place без идентификатора')

    ident_token = inner.children[0].label  # '"x"'
    name = strip_quotes(ident_token)
    return name

def contains_assignment(node: TreeViewNode) -> bool:
    """
    Проверяет, содержит ли дерево выражений операции присваивания.
    Присваивания должны быть только на верхнем уровне (как отдельные операторы).
    """
    if node is None:
        return False
    
    label = node.label
    # Проверяем, является ли текущий узел операцией присваивания
    if isinstance(label, str) and (label.startswith('store(') or label.startswith('store_at(')):
        return True
    
    # Рекурсивно проверяем всех детей
    for child in getattr(node, 'children', []) or []:
        if contains_assignment(child):
            return True
    
    return False

def parse_expr_list(node: TreeViewNode) -> list[TreeViewNode]:
    """
    list<expr>: (expr (',' expr)*)?
    В дереве могут быть только expr и запятые.
    Пустой список допускается.
    """
    # иногда генератор дерева делает label вроде 'list<expr> (empty)'
    if '(empty)' in node.label:
        return []

    if not node.children:
        return []

    result: list[TreeViewNode] = []
    for ch in node.children:
        if ch.label == 'expr':
            result.append(parse_expr(ch))
    return result


def make_leaf(label: str) -> TreeViewNode:
    """Создать лист нового дерева."""
    return TreeViewNode(label=label, node=None, children=[])

def parse_expr(expr: TreeViewNode) -> TreeViewNode:
    """
    expr
      └── <variant>   (place / literal / binary / unary / braces / indexer / call)
    """
    # как у тебя: expr → один реальный узел
    inner = expr.children[0]

    match inner.label:
        case 'place':
            return parse_place(inner)
        case 'literal':
            return parse_literal(inner)
        case 'braces':
            return parse_braces(inner)
        case 'unary':
            return parse_unary(inner)
        case 'binary':
            return parse_binary(inner)
        case 'indexer':
            return parse_indexer(inner)
        case 'call':
            return parse_call(inner)
        case _:
            raise SyntaxError(f'Неизвестный вид expr: {inner.label}')

def parse_place(node: TreeViewNode) -> TreeViewNode:
    # place
    #   └── "x"
    ident_token = node.children[0].label    # '"x"'
    name = strip_quotes(ident_token)
    return make_leaf(f'load({name})')

def parse_literal(node: TreeViewNode) -> TreeViewNode:
    # literal
    #   └── dec / bool / string ...
    type_node = node.children[0]           # например, 'dec'
    lit_type = type_node.label             # 'dec'

    value_token = type_node.children[0].label  # '"1"', '"false"', ...
    value = strip_quotes(value_token)

    return make_leaf(f'const({lit_type}({value}))')

def parse_braces(node: TreeViewNode) -> TreeViewNode:
    # braces
    #   ├── "("
    #   ├── expr
    #   └── ")"
    expr_node = node.children[1]
    return parse_expr(expr_node)

def parse_unary(node: TreeViewNode) -> TreeViewNode:
    # unary
    #   ├── unOp
    #   │   └── "!"
    #   └── expr
    op_token = node.children[0].children[0].label  # '"!"'
    op = strip_quotes(op_token)                    # '!'

    expr_node = node.children[1]  # expr
    child_ir = parse_expr(expr_node)
    
    # Запрещаем присваивания в операндах унарных операций
    if contains_assignment(child_ir):
        raise ValueError('Присваивание не может использоваться внутри выражения. Присваивания допускаются только как отдельные операторы.')

    return TreeViewNode(label=op, node=None, children=[child_ir])

def parse_binary(node: TreeViewNode) -> TreeViewNode:
    # binary
    #   ├── expr      (левый операнд)
    #   ├── binOp
    #   │   └── ":=" / "+" / "-" / "*" / ...
    #   └── expr      (правый операнд)

    left_expr  = node.children[0]
    op_node    = node.children[1]
    right_expr = node.children[2]

    op_token = op_node.children[0].label
    op = strip_quotes(op_token)

    # Спец-случай: присваивание
    if op == '=':
        rhs_ir = parse_expr(right_expr)
        
        # Запрещаем вложенные присваивания: правая часть не должна содержать присваиваний
        if contains_assignment(rhs_ir):
            raise ValueError('Присваивание не может использоваться внутри выражения. Присваивания допускаются только как отдельные операторы.')
        
        # Проверяем, что слева: place или indexer
        if left_expr.label != 'expr' or not left_expr.children:
            raise ValueError('Левая часть присваивания: ожидается expr')
        
        inner = left_expr.children[0]
        
        # Случай 1: простая переменная (place)
        if inner.label == 'place':
            name = strip_quotes(inner.children[0].label)
            return TreeViewNode(
                label=f'store({name})',
                node=None,
                children=[rhs_ir],
            )
        
        # Случай 2: индексатор (arr[idx] := val)
        if inner.label == 'indexer':
            # indexer: expr '[' list<expr> ']'
            base_expr = inner.children[0]
            list_node = inner.children[2]
            
            # База должна быть place
            base_name = extract_place_name_from_expr(
                base_expr,
                where='База индексатора в присваивании'
            )
            
            # Парсим индексы
            indices_ir = parse_expr_list(list_node)
            
            if not indices_ir:
                raise ValueError('Индексатор в присваивании без индексов')
            
            # Запрещаем присваивания в индексах присваивания
            for idx_ir in indices_ir:
                if contains_assignment(idx_ir):
                    raise ValueError('Присваивание не может использоваться внутри выражения. Присваивания допускаются только как отдельные операторы.')
            
            # Создаём store_at(arr) с детьми [idx, value]
            # Для многомерных массивов - последовательно
            # store_at(arr)[idx1, idx2, ..., value]
            return TreeViewNode(
                label=f'store_at({base_name})',
                node=None,
                children=[*indices_ir, rhs_ir],
            )
        
        raise ValueError(
            f'Левая часть присваивания: ожидается переменная или индексатор, '
            f'а не {inner.label!r}'
        )

    # Обычный бинарный оператор: +, -, *, /, ...
    left_ir = parse_expr(left_expr)
    right_ir = parse_expr(right_expr)
    
    # Запрещаем присваивания в операндах бинарных операций
    if contains_assignment(left_ir):
        raise ValueError('Присваивание не может использоваться внутри выражения. Присваивания допускаются только как отдельные операторы.')
    if contains_assignment(right_ir):
        raise ValueError('Присваивание не может использоваться внутри выражения. Присваивания допускаются только как отдельные операторы.')

    return TreeViewNode(
        label=op,   # '+', '-', '*', ...
        node=None,
        children=[left_ir, right_ir],
    )


def parse_indexer(node: TreeViewNode) -> TreeViewNode:
    # indexer
    #   ├── expr          (база: массив / вектор)
    #   ├── "["
    #   ├── list<expr>    (индексы)
    #   └── "]"

    base_expr = node.children[0]
    list_node = node.children[2]

    # 1) база индексатора должна быть place
    base_name = extract_place_name_from_expr(
        base_expr,
        where='База индексатора'
    )

    # 2) парсим индексы
    indices_ir = parse_expr_list(list_node)   # может быть []
    
    # Запрещаем присваивания в индексах
    for idx_ir in indices_ir:
        if contains_assignment(idx_ir):
            raise ValueError('Присваивание не может использоваться внутри выражения. Присваивания допускаются только как отдельные операторы.')

    # 3) создаём базу: load(a[])
    acc: TreeViewNode = make_leaf(f'load({base_name}[])')

    # 4) последовательно навешиваем операции индексации
    for idx_ir in indices_ir:
        acc = TreeViewNode(
            label='index',
            node=None,
            children=[acc, idx_ir],
        )

    return acc



def parse_call(node: TreeViewNode) -> TreeViewNode:
    # call
    #   ├── expr          (функция)
    #   ├── "("
    #   ├── list<expr>    (аргументы)
    #   └── ")"

    func_expr = node.children[0]
    args_list_node = node.children[2]

    # 1) проверяем, что имя функции — place
    func_name = extract_place_name_from_expr(
        func_expr,
        where='Имя функции в вызове'
    )

    # 2) парсим аргументы
    args_ir = parse_expr_list(args_list_node)
    
    # Запрещаем присваивания в аргументах функций
    for arg_ir in args_ir:
        if contains_assignment(arg_ir):
            raise ValueError('Присваивание не может использоваться внутри выражения. Присваивания допускаются только как отдельные операторы.')

    # IR: call(f) с дочерними узлами-аргументами
    return TreeViewNode(
        label=f'call({func_name})',
        node=None,
        children=args_ir,
    )

