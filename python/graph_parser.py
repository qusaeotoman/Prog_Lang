from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union

from tree_parser import TreeViewNode, tree_view_to_str
from graphviz import Digraph
from ast_generator import parse_expr

#######################################################################
# DATA STRUCT
#######################################################################

@dataclass
class Block:
    """
    Один блок графа (будущий прямоугольник на картинке).

    id      - уникальный номер блока
    label   - текст внутри блока (или можно класть сюда указатель на дерево операций)
    succs   - список исходящих рёбер вида (id_следующего_блока, метка_ребра)
              метка_ребра, например, "True"/"False" для if.
    """
    id: int
    label: str
    tree: TreeViewNode = field(default=None)
    succs: List[Tuple[int, Optional[str]]] = field(default_factory=list)


@dataclass
class CFG:
    """
    Весь граф потока управления.
    """
    blocks: Dict[int, Block] = field(default_factory=dict)
    next_id: int = 0  # счётчик для выдачи свежих id
    errors: List[str] = field(default_factory=list)
    call_names: set[str] = field(default_factory=set)

    def new_block(self, label: str, tree: TreeViewNode = None) -> Block:
        """
        Создаёт новый блок с данным текстом.
        """
        b = Block(self.next_id, label, tree)
        self.blocks[b.id] = b
        self.next_id += 1
        return b

    def add_edge(self, src: Block, dst: Block, label: Optional[str] = None):
        """
        Добавляет ребро src -> dst.
        label используется для подписей вроде True/False.
        """
        if dst is not None and src is not None:
            src.succs.append((dst.id, label))

    def remove_dangling_blocks(self, entry_id: int = 0, exit_id: int = 1) -> None:
        """Удаляет все блоки, недостижимые из entry_id, оставляя entry/exit."""
        reachable = set()
        stack = [entry_id]

        while stack:
            b_id = stack.pop()
            if b_id in reachable:
                continue
            block = self.blocks.get(b_id)
            if block is None:
                continue
            reachable.add(b_id)
            for succ_id, _ in block.succs:
                if succ_id not in reachable:
                    stack.append(succ_id)

        if exit_id in self.blocks:
            reachable.add(exit_id)

        for b_id in list(self.blocks.keys()):
            if b_id not in reachable:
                del self.blocks[b_id]

        for block in self.blocks.values():
            block.succs = [
                (dst_id, lbl)
                for (dst_id, lbl) in block.succs
                if dst_id in self.blocks
            ]

#######################################################################
# HELPERS
#######################################################################

def strip_quotes(s: str) -> str:
    if isinstance(s, str) and len(s) >= 2 and s[0] == s[-1] == '"':
        return s[1:-1]
    return s

def is_token(node: TreeViewNode, tok: str) -> bool:
    return node is not None and node.label in (tok, f'"{tok}"')

def dfs_find(node: TreeViewNode, label: str) -> List[TreeViewNode]:
    out = []
    if node is None:
        return out
    if node.label == label:
        out.append(node)
    for ch in getattr(node, "children", []) or []:
        out.extend(dfs_find(ch, label))
    return out

def first_child(node: TreeViewNode, label: str) -> Optional[TreeViewNode]:
    for ch in getattr(node, "children", []) or []:
        if ch.label == label:
            return ch
    return None

def collect_call_names(node: TreeViewNode, cfg: CFG) -> None:
    """Ищет операции call(name) внутри дерева операций."""
    if node is None:
        return

    label = node.label
    if isinstance(label, str) and label.startswith('call(') and label.endswith(')'):
        func_name = label[5:-1]
        cfg.call_names.add(func_name)

    for child in getattr(node, "children", []) or []:
        collect_call_names(child, cfg)

#######################################################################
# PARSER (Variant 1)
#######################################################################

def parse_expression(tree: TreeViewNode, graph: CFG, before: Block, label: str = None) -> Block:
    """
    Делает basic-block с деревом операций для expr.
    Работает как для узла expr, так и для узла expression (expr ';').
    """
    try:
        expr_node = None

        if tree.label == 'expr':
            expr_node = tree
        else:
            # часто expression -> [expr, ';'] или [expr]
            # ищем ребёнка expr
            expr_node = first_child(tree, 'expr')
            if expr_node is None and getattr(tree, "children", None):
                # fallback
                expr_node = tree.children[0]

        updated_tree = parse_expr(expr_node)
        collect_call_names(updated_tree, graph)

        expr_id = graph.new_block(tree_view_to_str(updated_tree), updated_tree)
    except ValueError as e:
        error_msg = str(e)
        graph.errors.append(error_msg)
        expr_id = graph.new_block(f"ERROR: {error_msg}")

    graph.add_edge(before, expr_id, label)
    return expr_id


def parse_var(tree: TreeViewNode, graph: CFG, before: Block, label: str = None) -> Block:
    """
    Variant 1 var statement:
      typeRef list<identifier ('=' expr)?> ';'

    Мы превращаем в дерево операций:
      var
        decl(x) / store(x)(rhs)
        ...
    """
    ops = []

    var_defs = dfs_find(tree, 'varDef')  # если grammar делает varDef
    if var_defs:
        for vd in var_defs:
            id_node = first_child(vd, 'identifier')
            if id_node is None:
                continue

            name_tok = id_node.children[0].label if id_node.children else id_node.label
            name = strip_quotes(name_tok)

            # initializer: ищем ребёнка expr или expression
            init = first_child(vd, 'expr')
            if init is None:
                init = first_child(vd, 'expression')

            if init is None:
                ops.append(TreeViewNode(label=f'decl({name})', node=None, children=[]))
            else:
                rhs_ir = parse_expr(init)
                collect_call_names(rhs_ir, graph)
                ops.append(TreeViewNode(label=f'store({name})', node=None, children=[rhs_ir]))
    else:
        # fallback: если varDef нет — просто создаём "var" без детализации
        ops.append(TreeViewNode(label='decl(?)', node=None, children=[]))

    var_tree = TreeViewNode(label='var', node=None, children=ops)
    var_block = graph.new_block(tree_view_to_str(var_tree), var_tree)
    graph.add_edge(before, var_block, label)
    return var_block


def parse_block(tree: TreeViewNode, graph: CFG, before: Block, label: str = None, end_cycle: Block = None) -> Block:
    """
    Variant 1 block:
      '{' statement* '}'
    """
    begin_id = graph.new_block("begin")
    end_id = graph.new_block("end")

    if before:
        graph.add_edge(before, begin_id, label)

    cur = begin_id

    # в tree.children могут быть '{', statement..., '}'
    for ch in getattr(tree, "children", []) or []:
        # пропускаем скобки
        if is_token(ch, '{') or is_token(ch, '}'):
            continue

        # ожидаем statement, но на всякий случай поддержим и прямые варианты
        cur = parse_statement(ch, graph, cur, None, end_cycle)

    graph.add_edge(cur, end_id)
    return end_id


def parse_if(tree: TreeViewNode, graph: CFG, before: Block, label: str = None, end_cycle: Block = None) -> Block:
    """
    Variant 1 if:
      'if' '(' expr ')' statement ('else' statement)?
    """
    # ищем expr и statements по позициям, но надёжно
    # children примерно: if, (, expr, ), statement, (else, statement)?
    expr_node = None
    stmts = []

    for ch in getattr(tree, "children", []) or []:
        if ch.label == 'expr' and expr_node is None:
            expr_node = ch
        if ch.label == 'statement':
            stmts.append(ch)

    if expr_node is None:
        # fallback: часто expr на позиции 2
        if len(tree.children) >= 3:
            expr_node = tree.children[2]

    expr_id = parse_expression(expr_node, graph, before, label)

    # then statement
    then_stmt = stmts[0] if len(stmts) >= 1 else (tree.children[4] if len(tree.children) >= 5 else None)
    then_end = parse_statement(then_stmt, graph, expr_id, 'True', end_cycle)

    end_if = graph.new_block('end_if')
    graph.add_edge(then_end, end_if)

    # else?
    has_else = any(is_token(ch, 'else') for ch in tree.children)
    if not has_else or len(stmts) < 2:
        graph.add_edge(expr_id, end_if, 'False')
    else:
        else_stmt = stmts[1]
        else_end = parse_statement(else_stmt, graph, expr_id, 'False', end_cycle)
        graph.add_edge(else_end, end_if)

    return end_if


def parse_while(tree: TreeViewNode, graph: CFG, before: Block, label: str = None) -> Block:
    """
    Variant 1 while:
      'while' '(' expr ')' statement
    """
    expr_node = None
    stmt_node = None

    # children: while, (, expr, ), statement
    for ch in getattr(tree, "children", []) or []:
        if ch.label == 'expr' and expr_node is None:
            expr_node = ch
        if ch.label == 'statement':
            stmt_node = ch

    if expr_node is None and len(tree.children) >= 3:
        expr_node = tree.children[2]
    if stmt_node is None and len(tree.children) >= 5:
        stmt_node = tree.children[4]

    expr_id = parse_expression(expr_node, graph, before, label)
    end_while = graph.new_block('end_while')

    body_end = parse_statement(stmt_node, graph, expr_id, 'True', end_while)

    graph.add_edge(body_end, expr_id)
    graph.add_edge(expr_id, end_while, 'False')
    return end_while


def parse_do(tree: TreeViewNode, graph: CFG, before: Block, label: str = None) -> Block:
    """
    Variant 1 do-while:
      'do' block 'while' '(' expr ')' ';'
    """
    start_do = graph.new_block('start_do')
    end_do = graph.new_block('end_do')

    graph.add_edge(before, start_do, label)

    # ищем block و expr
    block_node = first_child(tree, 'block')
    if block_node is None:
        # иногда block_content прямо تحت do
        block_node = first_child(tree, 'block_content')

    expr_node = None
    for ch in getattr(tree, "children", []) or []:
        if ch.label == 'expr':
            expr_node = ch
            break

    if expr_node is None:
        # غالباً expr على index 4
        if len(tree.children) >= 5:
            expr_node = tree.children[4]

    block_end = parse_block(block_node, graph, start_do, None, end_do)
    expr_id = parse_expression(expr_node, graph, block_end)

    graph.add_edge(expr_id, start_do, 'True')
    graph.add_edge(expr_id, end_do, 'False')
    return end_do


def parse_break(tree: TreeViewNode, graph: CFG, before: Block, label: str = None, end_cycle: Block = None) -> Optional[Block]:
    break_id = graph.new_block('break')
    graph.add_edge(before, break_id, label)

    if end_cycle is None:
        raise SyntaxError(f"Error: break without cycle at {tree.node.end_point if tree.node else 'unknown'}")

    graph.add_edge(break_id, end_cycle)
    return None


def parse_statement(tree: TreeViewNode, graph: CFG, before: Block, label: str = None, end_cycle: Block = None) -> Optional[Block]:
    """
    В проекте часто statement имеет одного ребёнка: [var|if|block|while|do|break|expression].
    لكن لو وصلنا مباشرة node من هذا النوع، نسانده.
    """
    inner = tree
    if tree is None:
        return before

    if tree.label == 'statement' and getattr(tree, "children", None):
        inner = tree.children[0]

    match inner.label:
        case 'block':
            return parse_block(inner, graph, before, label, end_cycle)
        case 'var':
            return parse_var(inner, graph, before, label)
        case 'expression':
            return parse_expression(inner, graph, before, label)
        case 'if':
            return parse_if(inner, graph, before, label, end_cycle)
        case 'while':
            return parse_while(inner, graph, before, label)
        case 'do':
            return parse_do(inner, graph, before, label)
        case 'break':
            return parse_break(inner, graph, before, label, end_cycle)
        case _:
            # إذا وصلنا node غير متوقع — حاول اعتباره expr
            if inner.label == 'expr':
                return parse_expression(inner, graph, before, label)
            print(inner.label)
            raise Exception("Такого блока нет")


def build_graph(tree: TreeViewNode) -> Tuple[CFG, List[str], List[str]]:
    """
    Variant 1 funcDef:
      funcSignature (block | ';')

    tree يمكن أن يكون: funcDef أو sourceItem أو غير ذلك.
    """
    cfg = CFG()

    # حاول الوصول لـ funcDef
    func_def = None
    if tree.label == 'funcDef':
        func_def = tree
    elif tree.label == 'sourceItem' and tree.children:
        func_def = tree.children[0]
    else:
        found = dfs_find(tree, 'funcDef')
        func_def = found[0] if found else None

    if func_def is None:
        return None, None, None

    # دالة إعلان فقط؟
    block_node = first_child(func_def, 'block')
    if block_node is None:
        # لا يوجد جسم
        return cfg, list(cfg.call_names), cfg.errors

    parse_block(block_node, cfg, None)
    return cfg, list(cfg.call_names), cfg.errors


# graph_parser.py

def get_func_name(node):
    """
    Extract function name from TreeViewNode produced by build_tree_view.

    Works for both shapes:
      sourceItem -> funcDef -> funcSignature -> name -> "foo"
    and cases where node is already funcDef/funcSignature.
    """

    def walk(n):
        if n is None:
            return
        yield n
        for ch in getattr(n, "children", []) or []:
            yield from walk(ch)

    def first_string_leaf(n):
        # leaf labels look like '"main"'
        lbl = getattr(n, "label", "")
        if isinstance(lbl, str) and len(lbl) >= 2 and lbl[0] == '"' and lbl[-1] == '"':
            return lbl.strip('"')
        for ch in getattr(n, "children", []) or []:
            s = first_string_leaf(ch)
            if s:
                return s
        return None

    # 1) best: look for a field node labeled 'name'
    for n in walk(node):
        if getattr(n, "label", None) == "name":
            s = first_string_leaf(n)
            if s:
                return s

    # 2) fallback: look for funcSignature then first quoted string after it
    in_sig = False
    for n in walk(node):
        if getattr(n, "label", None) == "funcSignature":
            in_sig = True
            continue
        if in_sig:
            s = first_string_leaf(n)
            if s:
                return s

    return "unknown"

#######################################################################
# RENDER
#######################################################################

def cfg_to_graphviz(cfg: CFG,
                    graph_name: str = "CFG",
                    node_shape: str = "box") -> Digraph:
    """
    Строит объект graphviz.Digraph по CFG.

    - Каждый блок рисуется прямоугольником (shape=node_shape) с label.
    - На рёбрах при наличии подписей используется label.
    """
    dot = Digraph(name=graph_name)
    dot.attr("node", shape=node_shape)

    for block in cfg.blocks.values():
        node_label = f"{block.label}"
        if block.tree is not None:
            lines = str(block.label).splitlines()
            node_label = ""
            for line in lines:
                node_label += line + "\\l"
        dot.node(str(block.id), label=node_label)

    for block in cfg.blocks.values():
        for succ_id, edge_label in block.succs:
            if succ_id not in cfg.blocks:
                continue
            edge_kwargs = {}
            if edge_label is not None:
                edge_kwargs["label"] = edge_label
            dot.edge(str(block.id), str(succ_id), **edge_kwargs)

    return dot


def render_cfg(cfg: CFG,
               filename: str = "cfg",
               fmt: str = "svg") -> None:
    """
    Рисует CFG в файл (по умолчанию cfg.png) с помощью graphviz.

    :param cfg: объект CFG
    :param filename: имя файла без расширения
    :param fmt: формат (png, pdf, svg, ...)
    """
    if cfg is None:
        return
    dot = cfg_to_graphviz(cfg)
    dot.render(filename, format=fmt, cleanup=True)
