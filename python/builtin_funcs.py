def generate_alloc(out_file):
    """Генерирует код для функции alloc."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
alloc:
     ; frame layout: [bp] saved bp, [bp+4] return, [bp+8] size_shift, [bp+12] len
     ldbp 12       ; push len
     ldbp 8        ; push size_shift
     shl           ; total_bytes = len << size_shift
     pushhp        ; old hp -> will be return value
     stbp 8        ; store return value at [bp+8], pop old hp
     addhp         ; hp = hp + total_bytes, pop total
     ret
""")


def _generate_type_constructor(out_file, type_name, size_shift):
    """
    Общая функция для генерации конструктора типа.
    
    Args:
        out_file: Путь к выходному файлу
        type_name: Имя типа (bool, byte, int, etc.)
        size_shift: Сдвиг размера для alloc (0 для 1 байта, 1 для 2 байт, 2 для 4 байт)
    """
    # Выравнивание комментариев: название функции до колонки 20, push до колонки 20
    label_col = 24
    push_col = 24
    indent = "     "  # 5 пробелов для отступа команд
    
    # Форматируем название функции с выравниванием комментария
    label_line = f"{type_name}:"
    label_padding = " " * (label_col - len(label_line))
    label_line = f"{label_line}{label_padding}; Builtin constructor: {type_name}(size) -> array[] of {type_name}"
    
    # Форматируем push с выравниванием комментария
    # Учитываем отступ (5 пробелов) + команда push + число
    push_cmd = f"push {size_shift}"
    total_push_len = len(indent) + len(push_cmd)
    push_padding = " " * (push_col - total_push_len)
    push_line = f"{indent}{push_cmd}{push_padding}; push size_shift to stack"
    
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write(f"""
{label_line}
     ldbp 8             ; load len (size of array) from [bp+8] and push to stack
{push_line}
     call alloc         ; alloc(size_shift, len) -> result on TOS after ret
     stbp 8             ; store result at [bp+8] and pop
     ret
""")


def generate_read_byte(out_file): # Вызывать с 1 аргументом для результата
    """Генерирует код для встроенной функции read_byte."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
read_byte:      ; Builtin function: read_byte()
    inb         ; read in byte
    stbp 8      ; store at bp + 8
    ret
""")

def generate_send_byte(out_file):
    """Генерирует код для встроенной функции send_byte."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
send_byte:      ; Builtin function: send_byte(b: byte)
    ldbp 8      ; load bp + 8
    outb        ; send byte
    ret
""")


def generate_bool_constructor(out_file):
    """Генерирует код для конструктора типа bool (1 байт, size_shift=0)."""
    _generate_type_constructor(out_file, 'bool', 0)


def generate_byte_constructor(out_file):
    """Генерирует код для конструктора типа byte (1 байт, size_shift=0)."""
    _generate_type_constructor(out_file, 'byte', 0)


def generate_int_constructor(out_file):
    """Генерирует код для конструктора типа int (2 байта, size_shift=1)."""
    _generate_type_constructor(out_file, 'int', 1)


def generate_uint_constructor(out_file):
    """Генерирует код для конструктора типа uint (2 байта, size_shift=1)."""
    _generate_type_constructor(out_file, 'uint', 1)


def generate_long_constructor(out_file):
    """Генерирует код для конструктора типа long (4 байта, size_shift=2)."""
    _generate_type_constructor(out_file, 'long', 2)


def generate_ulong_constructor(out_file):
    """Генерирует код для конструктора типа ulong (4 байта, size_shift=2)."""
    _generate_type_constructor(out_file, 'ulong', 2)


def generate_char_constructor(out_file):
    """Генерирует код для конструктора типа char (1 байт, size_shift=0)."""
    _generate_type_constructor(out_file, 'char', 0)


def generate_bool_to_byte(out_file):
    """Генерирует код для встроенной функции bool_to_byte."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
bool_to_byte:    ; Builtin function: bool_to_byte(b: bool) -> byte
    ret          ; value already at [bp+8], no conversion needed
""")


def generate_byte_to_bool(out_file):
    """Генерирует код для встроенной функции byte_to_bool."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
byte_to_bool:    ; Builtin function: byte_to_bool(b: byte) -> bool
    ret          ; value already at [bp+8], no conversion needed
""")


def generate_byte_to_int(out_file):
    """Генерирует код для встроенной функции byte_to_int."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
byte_to_int:     ; Builtin function: byte_to_int(b: byte) -> int
    ret          ; value already at [bp+8], already zero-extended to 32 bits
""")


def generate_int_to_byte(out_file):
    """Генерирует код для встроенной функции int_to_byte."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
int_to_byte:     ; Builtin function: int_to_byte(i: int) -> byte
    ldbp 8       ; load int value from [bp+8]
    push 0xFF    ; mask for byte (8 bits)
    band         ; apply mask
    stbp 8       ; store result at [bp+8]
    ret
""")


def generate_int_to_uint(out_file):
    """Генерирует код для встроенной функции int_to_uint."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
int_to_uint:     ; Builtin function: int_to_uint(i: int) -> uint
    ret          ; value already at [bp+8], no conversion needed
""")


def generate_uint_to_int(out_file):
    """Генерирует код для встроенной функции uint_to_int."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
uint_to_int:     ; Builtin function: uint_to_int(u: uint) -> int
    ret          ; value already at [bp+8], no conversion needed
""")


def generate_int_to_long(out_file):
    """Генерирует код для встроенной функции int_to_long."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
int_to_long:     ; Builtin function: int_to_long(i: int) -> long
    ldbp 8       ; load int value from [bp+8]
    push 0xFFFF  ; mask to get only lower 16 bits
    band         ; keep only lower 16 bits (clears upper 16 bits)
    push 0x8000  ; sign bit mask
    bxor         ; flip sign bit: (x ^ 0x8000)
    push 0x8000  ; prepare for subtraction
    sub          ; (x ^ 0x8000) - 0x8000 (sign extend 16->32)
    stbp 8       ; store result at [bp+8]
    ret
""")


def generate_long_to_int(out_file):
    """Генерирует код для встроенной функции long_to_int."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
long_to_int:     ; Builtin function: long_to_int(l: long) -> int
    ldbp 8       ; load long value from [bp+8]
    push 0xFFFF  ; mask for int (16 bits)
    band         ; apply mask to get lower 16 bits
    stbp 8       ; store result at [bp+8]
    ret
""")


def generate_long_to_ulong(out_file):
    """Генерирует код для встроенной функции long_to_ulong."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
long_to_ulong:   ; Builtin function: long_to_ulong(l: long) -> ulong
    ret          ; value already at [bp+8], no conversion needed
""")


def generate_ulong_to_long(out_file):
    """Генерирует код для встроенной функции ulong_to_long."""
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write("""
ulong_to_long:   ; Builtin function: ulong_to_long(u: ulong) -> long
    ret          ; value already at [bp+8], no conversion needed
""")