import argparse
import os
import subprocess
from ctypes import CDLL, PYFUNCTYPE, pythonapi, c_void_p, c_char_p, py_object
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter

def parse_cli():
    """
    Поддерживаются два режима запуска:

    1) Уже собранная библиотека:
       python main.py --lib path/to/parser.dll [--lang foo] input.txt output.txt

       --lib  : путь к .dll/.so/.dylib
       --lang : имя языка (foo => функция tree_sitter_foo). Если не указано,
                имя берётся из имени файла библиотеки.

    2) Грамматика-проект (нужно собрать библиотеку):
       python main.py path/to/grammar foo input.txt output.txt

       grammar_dir : путь к папке с grammar.js
       lang_name   : имя языка (foo => функция tree_sitter_foo)
    """
    p = argparse.ArgumentParser(description="Сборка и запуск tree-sitter парсера")

    # Режим 1: уже собранная библиотека
    p.add_argument(
        "--lib",
        dest="lib_path",
        help=(
            "Путь к уже скомпилированной tree-sitter библиотеке (.so/.dll/.dylib). "
            "В этом режиме grammar_dir и lang_name позиционные не используются."
        ),
    )
    p.add_argument(
        "--lang",
        dest="lib_lang_name",
        help=(
            "Имя языка при использовании --lib (например, 'foo' для tree_sitter_foo). "
            "Если не указано, будет выведено из имени файла библиотеки."
        ),
    )

    # Режим 2: проект с грамматикой (сборка)
    p.add_argument("grammar_dir", nargs="?", help="Путь к папке с грамматикой (library)")
    p.add_argument(
        "lang_name",
        nargs="?",
        help="Имя парсера, напр. 'foo' для функции tree_sitter_foo",
    )

    # Общие аргументы для обоих режимов
    p.add_argument("file_path", help="Путь к файлу для парсинга")
    p.add_argument("out_file_path", help="Путь к выходному файлу")

    a = p.parse_args()

    using_lib = a.lib_path is not None
    using_grammar = a.grammar_dir is not None or a.lang_name is not None

    # Защита от одновременного указания двух режимов
    if using_lib and using_grammar:
        p.error(
            "Нельзя одновременно указывать --lib и grammar_dir/lang_name.\n"
            "Выберите один из двух вариантов:\n"
            "  1) --lib path/to/lib [--lang foo] file_path out_file_path\n"
            "  2) grammar_dir lang_name file_path out_file_path"
        )

    # Режим 1: уже собранная библиотека
    if using_lib:
        grammar_dir = None
        lib_path = a.lib_path

        # Имя языка: сначала берём из --lang, иначе вытаскиваем из имени файла
        if a.lib_lang_name:
            lang_name = a.lib_lang_name
        else:
            stem = Path(a.lib_path).stem  # e.g. libfoo -> libfoo, foo -> foo
            lang_name = stem[3:] if stem.startswith("lib") else stem

    # Режим 2: сборка из грамматики
    else:
        if not a.grammar_dir or not a.lang_name:
            p.error(
                "В режиме сборки из грамматики нужно указать grammar_dir и lang_name.\n"
                "Пример: python main.py path/to/grammar foo input.txt output.txt"
            )
        grammar_dir = a.grammar_dir
        lang_name = a.lang_name
        lib_path = None

    return grammar_dir, lang_name, a.file_path, a.out_file_path, lib_path

def build_parser(grammar_dir: str, lang_name: str) -> str:
    """Собрать shared-библиотеку в <grammar_dir>/build/<lang_name>.(dll|so|dylib)."""
    out_dir = os.path.join(grammar_dir, "build")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lang_name}.dll")

    subprocess.run(
        ["tree-sitter", "generate", "--abi", str(tree_sitter.LANGUAGE_VERSION)],
        cwd=grammar_dir,
        check=True,
    )
    subprocess.run(
        ["tree-sitter", "build", "-o", out_path], cwd=grammar_dir, check=True
    )
    return out_path


def load_and_parse(lib_path: str, lang_name: str, file_path: str):
    """Загрузить язык из DLL, распарсить файл, вернуть корневой узел."""
    cdll = CDLL(os.path.abspath(lib_path))
    func_name = f"tree_sitter_{lang_name}"
    if not hasattr(cdll, func_name) and hasattr(cdll, func_name + "_language"):
        func_name = func_name + "_language"
    func = getattr(cdll, func_name)
    func.restype = c_void_p
    ptr = func()

    PyCapsule_New = PYFUNCTYPE(py_object, c_void_p, c_char_p, c_void_p)(
        ("PyCapsule_New", pythonapi)
    )
    capsule = PyCapsule_New(ptr, b"tree_sitter.Language", None)
    lang = Language(capsule)

    parser = Parser(lang)
    with open(file_path, "rb") as f:
        data = f.read()
    tree = parser.parse(data, encoding="utf8")
    return tree.root_node

def get_tree_root(lib_path, lang_name, file_path, grammar_dir):
    if lib_path:
        root = load_and_parse(lib_path, lang_name, file_path)
    else:
        built_lib = build_parser(grammar_dir, lang_name)
        root = load_and_parse(built_lib, lang_name, file_path)
    return root 