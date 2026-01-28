from file_parser_to_graph import (
    parse_cli,
    analyze_files,
    render_call_graph,
    write_errors_report,
)
from generate_asm import generate_asm
from types_generator import process_type, check_main_function
from type_checker import render_all_typed_cfgs
from pathlib import Path


def write_type_errors(errors, out_dir_path: Path):
    type_errors_path = out_dir_path / "type_errors.txt"
    with open(type_errors_path, "w", encoding="utf-8") as f:
        for func_name, err_list in errors.items():
            f.write(f"=== {func_name} ===\n")
            for err in err_list:
                f.write(f"{err}\n")
            f.write("\n")
    print(f"Ошибки типизации записаны в {type_errors_path}")


def render_typed_graphs(typed_data, out_dir_path: Path):
    typed_graph_dir = out_dir_path / "typed_graph"
    render_all_typed_cfgs(typed_data, str(typed_graph_dir), fmt="svg")


def handle_type_check(result, out_dir_path: Path) -> tuple[bool, dict, dict]:
    # process_type возвращает typed_data - та же структура, но с типами в node.type
    # и funcs_returns - информацию о возвращаемых типах функций
    typed_data, errors, funcs_returns = process_type(result)

    if errors:
        write_type_errors(errors, out_dir_path)
        return True, None, None  # Ошибки найдены, останавливаем трансляцию

    if typed_data:
        render_typed_graphs(typed_data, out_dir_path)
    
    return False, typed_data, funcs_returns  # Ошибок нет, продолжаем трансляцию


def main():
    grammar_dir, lang_name, file_paths, out_dir, lib_path = parse_cli()
    result = analyze_files(file_paths, lib_path, lang_name, grammar_dir, out_dir=out_dir)
    
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    call_graph_base = out_dir_path / "call_graph"
    render_call_graph(result, filename=str(call_graph_base), fmt="svg")

    errors_report_path = call_graph_base.with_suffix(".errors.txt")
    ready_assemble = write_errors_report(result, filename=str(errors_report_path))
    
    if ready_assemble:
        return

    # Проверка типов: если есть ошибки, останавливаем трансляцию
    has_errors, typed_data, funcs_returns = handle_type_check(result, out_dir_path)
    if has_errors:
        return
    
    # Проверка наличия функции main без аргументов и без возвращаемого значения
    if not check_main_function(result, errors_report_path):
        return
    
    # Проверяем, что typed_data не None перед использованием
    if typed_data is None:
        print("Ошибка: typed_data равен None")
        return
    
    asm_file = out_dir_path / "result.asm"
    
    # Передаем typed_data (с обновленными vars) и funcs_returns в generate_asm
    generate_asm(typed_data, str(asm_file), funcs_returns)
    print(f"Ассемблерный код сохранен в {asm_file}")

    return

if __name__ == "__main__":
    main()
