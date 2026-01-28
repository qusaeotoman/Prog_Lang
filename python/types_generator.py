from file_parser_to_graph import BUILTIN_TYPES, BUILTIN_FUNCTIONS
from type_checker import (
    TypeChecker,
    check_all_functions,
    format_type_errors,
    normalize_type,
)

def get_type_from_typeRef(type_ref_node):
    """
    type_ref_node — узел TreeViewNode(label='typeRef', ...)

    Возвращает кортеж:
      (type_str, error_msg)

    type_str:
      - строка вида "int", "MyType", "array[] of int" и т.п.
      - None, если тип не удалось корректно определить

    error_msg:
      - None, если ошибок нет
      - строка с описанием ошибки (например, про многомерный массив или custom)
    """
    kind = type_ref_node.children[0]

    # builtin: 'int', 'bool', 'string', ...
    if kind.label == 'builtin':
        token = kind.children[0]
        return normalize_type(token.label.strip('"')), None

    # custom: identifier — считаем ошибкой (классы не поддерживаются)
    # custom сам является identifier (не содержит вложенный identifier)
    if kind.label == 'custom':
        token = kind.children[0]
        type_name = token.label.strip('"')
        return type_name, "классы не поддерживаются"

    # иногда грамматика может давать сразу identifier
    if kind.label == 'identifier':
        token = kind.children[0]
        # это тоже пользовательский тип
        type_name = token.label.strip('"')
        return type_name, "классы не поддерживаются"

    # array: 'array' '[' (',')* ']' 'of' typeRef
    if kind.label == 'array':
        # проверяем количество запятых в квадратных скобках
        comma_count = sum(1 for c in kind.children if c.label == '\",\"')
        if comma_count > 0:
            return None, "многомерные массивы не поддерживаются"

        # находим вложенный typeRef (тип элементов массива)
        inner_type_ref = next(c for c in kind.children if c.label == 'typeRef')
        inner_kind = inner_type_ref.children[0]

        # array of array — считаем многомерным
        if inner_kind.label == 'array':
            return None, "многомерные массивы не поддерживаются"

        # array of string — тоже запрещаем (string трактуем как массив символов)
        if inner_kind.label == 'builtin':
            inner_token = inner_kind.children[0].label.strip('"').lower()
            if inner_token == 'string':
                return None, "массивы строк не поддерживаются"

        inner_str, inner_err = get_type_from_typeRef(inner_type_ref)
        if inner_err is not None:
            return None, inner_err
        inner_str = normalize_type(inner_str)

        return normalize_type(f"array[] of {inner_str}"), None

    return None, f"неподдерживаемый вид typeRef: {kind.label}"


def get_dict_var(nodes):
    """
    nodes — это последовательность узлов:
    list<identifier> ':' typeRef? ';' list<identifier> ':' typeRef? ';' ...

    Возвращает:
      types_dict: { var_name: (type_str | None, type_ref_node | None) }
      errors:     список строк с описанием ошибок
    """
    nodes = list(nodes)
    i = 0
    types_dict: dict[str, tuple[str | None, object | None]] = {}
    errors: list[str] = []

    while i < len(nodes):
        node = nodes[i]

        # ожидаем list<identifier>
        if node.label != 'list<identifier>':
            i += 1
            continue

        # собираем имена идентификаторов a,b,c,...
        names = [
            ident.children[0].label.strip('"')
            for ident in node.children
            if ident.label == 'identifier'
        ]

        type_node = None
        type_str = None
        error_for_this_decl = None

        # смотрим, есть ли дальше ':' typeRef
        j = i + 1
        if j < len(nodes) and nodes[j].label == '":"':
            j += 1
            if j < len(nodes) and nodes[j].label == 'typeRef':
                type_node = nodes[j]
                type_str, type_err = get_type_from_typeRef(type_node)
                if type_err is not None:
                    error_for_this_decl = type_err
            else:
                error_for_this_decl = "ожидался typeRef после ':'"
        else:
            error_for_this_decl = "тип не задан"

        # записываем результат для всех имён
        for name in names:
            if name in types_dict:
                errors.append(f"повторное объявление переменной '{name}'")
                continue
            types_dict[name] = (type_str, type_node)

        if error_for_this_decl is not None:
            errors.append(
                f"{error_for_this_decl} (переменные: {', '.join(names)})"
            )

        # сдвигаем i к следующему объявлению: проматываем до ';'
        i = j
        while i < len(nodes) and nodes[i].label != '";"':
            i += 1
        if i < len(nodes) and nodes[i].label == '";"':
            i += 1

    return types_dict, errors


def get_func_returns_type(tree):
    """
    tree — узел функции.
    Возвращает:
      (type_str | None, type_ref_node | None, error_msg | None)
    """
    # Проверяем структуру дерева перед доступом
    if not tree.children or len(tree.children) < 1:
        return (None, None), "Неожиданная структура дерева функции: отсутствуют дочерние узлы"
    
    func_def_wrapper = tree.children[0]
    if not func_def_wrapper.children or len(func_def_wrapper.children) < 2:
        return (None, None), "Неожиданная структура дерева функции: отсутствует funcSignature"
    
    func_signature = func_def_wrapper.children[1]
    if not func_signature.children:
        return (None, None), "Неожиданная структура дерева функции: funcSignature пуст"
    
    type_ref_node = func_signature.children[-1]

    # тип не задан (процедура)
    if type_ref_node.label != 'typeRef':
        return (None, None), None

    type_str, error = get_type_from_typeRef(type_ref_node)

    # если тип разобрать не удалось или он некорректен
    if error:
        # узел typeRef всё равно возвращаем, чтобы можно было повесить диагностику
        return (None, type_ref_node), error

    # всё ок: строка типа + сам узел typeRef
    return (type_str, type_ref_node), None


def get_args_dict(arg_nodes):
    """
    arg_nodes — список вида:
      [argDef, ",", argDef, ",", argDef, ...]
    Каждый argDef: identifier (':' typeRef)?

    Возвращает:
      args_dict: { arg_name: (type_str | None, type_ref_node | None) }
      errors:    список строк с ошибками
    """
    args_dict: dict[str, tuple[str | None, object | None]] = {}
    errors: list[str] = []

    # Берём только узлы argDef, запятые игнорируем
    for arg in arg_nodes:
        if arg.label != 'argDef':
            continue

        # 1. имя аргумента
        ident_node = next(c for c in arg.children if c.label == 'identifier')
        name = ident_node.children[0].label.strip('"')

        type_node = None
        type_str = None
        error_msg = None

        # 2. ищем ':' и typeRef внутри argDef
        has_colon = any(c.label == '":"'
                        for c in arg.children)
        if has_colon:
            type_child = [c for c in arg.children if c.label == 'typeRef']
            if type_child:
                type_node = type_child[0]
                type_str, type_err = get_type_from_typeRef(type_node)
                if type_err is not None:
                    error_msg = type_err
            else:
                error_msg = "ожидался typeRef после ':'"
        else:
            error_msg = "тип не задан"

        # 3. проверка на дубликат аргумента
        if name in args_dict:
            errors.append(f"повторное объявление аргумента '{name}'")
        else:
            args_dict[name] = (type_str, type_node)

        if error_msg is not None:
            errors.append(f"{error_msg} (аргумент: {name})")

    return args_dict, errors

def check_var_param_conflicts(funcs_vars, funcs_calls):
    """
    funcs_vars: { func_name: { var_name: (...) } }
    funcs_calls: { func_name: { arg_name: (...) } }

    Возвращает:
      { func_name: [строки_ошибок] }
    """
    conflicts = {}

    for func_name, vars_dict in funcs_vars.items():
        params_dict = funcs_calls.get(func_name, {}) or {}
        var_names = set(vars_dict.keys())
        param_names = set(params_dict.keys())

        intersect = var_names & param_names
        if intersect:
            names_str = ", ".join(sorted(intersect))
            msg = (
                f"имена параметров и локальных переменных совпадают: {names_str}"
            )
            conflicts[func_name] = [msg]

    return conflicts

def process_type(not_typed_data: dict):
    global_errors = {}
    funcs_returns = {}
    funcs_calls = {}
    funcs_vars = {}
    # Типы данных функций
    for func_name, (references, _, cfg, tree, params, vars) in not_typed_data.items():
        # Пропускаем псевдо-узлы файлов
        if func_name.startswith('<file:'):
            continue
        
        # Пропускаем функции без тела (только объявления)
        if cfg is None:
            continue
            
        # Тип данных, возвращаемый функцией
        func_type, error = get_func_returns_type(tree)
        if error:
            global_errors[func_name] = [error]
        funcs_returns[func_name] = func_type
        
        # Типы данных локальных переменных функции
        # tree -> funcDef -> [method, funcSignature, body]
        if not tree.children or len(tree.children) < 1:
            global_errors[func_name] = global_errors.get(func_name, []) + ["Неожиданная структура дерева функции"]
            continue
        
        func_def = tree.children[0]
        if not func_def.children or len(func_def.children) < 2:
            global_errors[func_name] = global_errors.get(func_name, []) + ["Неожиданная структура дерева функции: отсутствует funcSignature"]
            continue
        
        body_node = func_def.children[2] if len(func_def.children) > 2 else None
        
        if body_node and body_node.label == 'body':
            # body -> [var?, list<var_decl>, block]
            # Переменные находятся между первым элементом и последним (block)
            types = body_node.children[1:-1]
            dict_var, error = get_dict_var(types)
            if error:
                global_errors[func_name] = global_errors.get(func_name, []) + error
            funcs_vars[func_name] = dict_var
        else:
            funcs_vars[func_name] = {}
        
        # Если функция возвращает значение, добавляем переменную с именем функции
        # в конец локальных переменных для проверки конфликтов
        if func_type[0] is not None:
            funcs_vars[func_name][func_name] = (func_type[0], None)
        
        # Типы данных аргументов функции
        func_signature = func_def.children[1]
        # funcSignature -> [identifier, '(', list_argDef?, ')', (':' typeRef)?]
        arg_list_node = None
        for child in func_signature.children:
            if child.label == 'list<argDef>':
                arg_list_node = child
                break
        
        if arg_list_node:
            func_call, error = get_args_dict(arg_list_node.children)
        else:
            func_call, error = {}, []
            
        if error:
            global_errors[func_name] = global_errors.get(func_name, []) + error
        funcs_calls[func_name] = func_call
    
    # Проверяем, что пользователь не объявил функции с именами встроенных
    user_func_names = {
        name for name in funcs_returns.keys() if not name.startswith('<file:')
    }
    reserved_conflicts = sorted(user_func_names & BUILTIN_FUNCTIONS)
    if reserved_conflicts:
        for name in reserved_conflicts:
            msg = (
                f"имя функции '{name}' зарезервировано для встроенной функции "
                "и не может быть переопределено"
            )
            global_errors[name] = global_errors.get(name, []) + [msg]

    # Если есть ошибки, вернем
    if global_errors:
        return None, global_errors, None
    
    # Проверим конфликты
    conflicts = check_var_param_conflicts(funcs_vars, funcs_calls)
    if conflicts:
        for func_name, msgs in conflicts.items():
            global_errors[func_name] = global_errors.get(func_name, []) + msgs
        return None, global_errors, None
    
    # Добавляем встроенные функции в funcs_returns и funcs_calls, 
    # 1. read_byte(): byte
    funcs_returns['read_byte'] = ('byte', None)
    funcs_calls['read_byte'] = {}

    # 2. send_byte(b: byte): void
    funcs_returns['send_byte'] = (None, None)
    funcs_calls['send_byte'] = {'b': ('byte', None)}

    # 3. Type conversion functions
    funcs_returns['bool_to_byte'] = ('byte', None)
    funcs_calls['bool_to_byte'] = {'b': ('bool', None)}
    
    funcs_returns['byte_to_bool'] = ('bool', None)
    funcs_calls['byte_to_bool'] = {'b': ('byte', None)}
    
    funcs_returns['byte_to_int'] = ('int', None)
    funcs_calls['byte_to_int'] = {'b': ('byte', None)}
    
    funcs_returns['int_to_byte'] = ('byte', None)
    funcs_calls['int_to_byte'] = {'i': ('int', None)}
    
    funcs_returns['int_to_uint'] = ('uint', None)
    funcs_calls['int_to_uint'] = {'i': ('int', None)}
    
    funcs_returns['uint_to_int'] = ('int', None)
    funcs_calls['uint_to_int'] = {'u': ('uint', None)}
    
    funcs_returns['int_to_long'] = ('long', None)
    funcs_calls['int_to_long'] = {'i': ('int', None)}
    
    funcs_returns['long_to_int'] = ('int', None)
    funcs_calls['long_to_int'] = {'l': ('long', None)}
    
    funcs_returns['long_to_ulong'] = ('ulong', None)
    funcs_calls['long_to_ulong'] = {'l': ('long', None)}
    
    funcs_returns['ulong_to_long'] = ('long', None)
    funcs_calls['ulong_to_long'] = {'u': ('ulong', None)}

    # 4. Конструкторы массивов: int(size) -> array[] of int, и т.д.
    for t_name in BUILTIN_TYPES:
        # returns: array[] of <t_name> (char -> byte, string -> array of byte)
        normalized_name = normalize_type(t_name)
        ret_type = normalize_type(f"array[] of {normalized_name}")
        funcs_returns[t_name] = (ret_type, None)
        
        # args: size: int
        funcs_calls[t_name] = {'size': ('int', None)}

    # Типы данных в листьях: проверяем типы во всех выражениях
    # После этого вызова в not_typed_data узлы будут иметь заполненные node.type
    type_check_results = check_all_functions(
        not_typed_data,
        funcs_vars,
        funcs_calls,
        funcs_returns,
    )
    
    # Собираем ошибки типизации
    type_errors = format_type_errors(type_check_results)
    for func_name, errors in type_errors.items():
        global_errors[func_name] = global_errors.get(func_name, []) + errors
    
    if global_errors:
        return None, global_errors, None
    
    # Обновляем структуру данных: добавляем переменную с именем функции в список vars
    # для функций с возвращаемым значением
    typed_data = {}
    for func_name, (references, errors, cfg, tree, params, vars) in not_typed_data.items():
        # Если функция возвращает значение, добавляем переменную с именем функции в конец vars
        func_type = funcs_returns.get(func_name)
        if func_type and func_type[0] is not None:
            # Создаем новый список vars с добавленной переменной
            vars = list(vars)  # Копируем список
            vars.append((func_name, func_type[0]))  # Добавляем переменную с именем функции
        typed_data[func_name] = (references, errors, cfg, tree, params, vars)
    
    # Возвращаем typed_data - это та же структура, но с заполненными типами
    # not_typed_data теперь типизирована (node.type заполнены)
    # Также возвращаем funcs_returns для использования в generate_asm
    return typed_data, global_errors, funcs_returns


def check_main_function(result, errors_report_path) -> bool:
    """
    Проверяет наличие функции main без аргументов и без возвращаемого значения.
    Возвращает True, если функция main найдена и соответствует требованиям.
    """
    from pathlib import Path
    
    main_func = None
    for func_name, (calls, errors, cfg, tree, params, vars) in result.items():
        # Пропускаем псевдо-узлы файлов
        if func_name.startswith('<file:'):
            continue
        
        # Пропускаем функции без тела
        if cfg is None:
            continue
        
        if func_name == 'main':
            main_func = (func_name, tree)
            break
    
    if main_func is None:
        # Записываем ошибку в файл с логами
        errors_report_path = Path(errors_report_path)
        with open(errors_report_path, "a", encoding="utf-8") as f:
            f.write("\n=== MAIN FUNCTION ERROR ===\n")
            f.write("Функция main без аргументов и без возвращаемого значения не найдена\n")
        print(f"Ошибка: функция main не найдена. Детали в {errors_report_path}")
        return False
    
    func_name, tree = main_func
    
    # Проверяем возвращаемый тип (должен быть None - процедура)
    func_type, error = get_func_returns_type(tree)
    if error or func_type[0] is not None:
        errors_report_path = Path(errors_report_path)
        with open(errors_report_path, "a", encoding="utf-8") as f:
            f.write("\n=== MAIN FUNCTION ERROR ===\n")
            if error:
                f.write(f"Ошибка при проверке возвращаемого типа функции main: {error}\n")
            else:
                f.write(f"Функция main имеет возвращаемый тип: {func_type[0]}, ожидается void (без возвращаемого значения)\n")
        print(f"Ошибка: функция main имеет возвращаемое значение. Детали в {errors_report_path}")
        return False
    
    # Проверяем аргументы (должно быть пусто)
    func_def = tree.children[0]
    func_signature = func_def.children[1]
    # funcSignature -> [identifier, '(', list_argDef?, ')', (':' typeRef)?]
    arg_list_node = None
    for child in func_signature.children:
        if child.label == 'list<argDef>':
            arg_list_node = child
            break
    
    if arg_list_node:
        func_args, _ = get_args_dict(arg_list_node.children)
        if func_args:
            errors_report_path = Path(errors_report_path)
            with open(errors_report_path, "a", encoding="utf-8") as f:
                f.write("\n=== MAIN FUNCTION ERROR ===\n")
                f.write(f"Функция main имеет аргументы: {', '.join(func_args.keys())}, ожидается функция без аргументов\n")
            print(f"Ошибка: функция main имеет аргументы. Детали в {errors_report_path}")
            return False
    
    return True