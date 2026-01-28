[section code, code]
ldsp 0xFFFC      ; even-aligned stack top
ldhp 0x0000      ; init heap base
setbp            ; establish caller bp before first call
call main
hlt

read_byte:      ; Builtin function: read_byte()
    inb         ; read in byte
    stbp 8      ; store at bp + 8
    ret

send_byte:      ; Builtin function: send_byte(b: byte)
    ldbp 8      ; load bp + 8
    outb        ; send byte
    ret

alloc:
     ; frame layout: [bp] saved bp, [bp+4] return, [bp+8] size_shift, [bp+12] len
     ldbp 12       ; push len
     ldbp 8        ; push size_shift
     shl           ; total_bytes = len << size_shift
     pushhp        ; old hp -> will be return value
     stbp 8        ; store return value at [bp+8], pop old hp
     addhp         ; hp = hp + total_bytes, pop total
     ret

bool:                   ; Builtin constructor: bool(size) -> array[] of bool
     ldbp 8             ; load len (size of array) from [bp+8] and push to stack
     push 0             ; push size_shift to stack
     call alloc         ; alloc(size_shift, len) -> result on TOS after ret
     stbp 8             ; store result at [bp+8] and pop
     ret

byte:                   ; Builtin constructor: byte(size) -> array[] of byte
     ldbp 8             ; load len (size of array) from [bp+8] and push to stack
     push 0             ; push size_shift to stack
     call alloc         ; alloc(size_shift, len) -> result on TOS after ret
     stbp 8             ; store result at [bp+8] and pop
     ret

int:                    ; Builtin constructor: int(size) -> array[] of int
     ldbp 8             ; load len (size of array) from [bp+8] and push to stack
     push 1             ; push size_shift to stack
     call alloc         ; alloc(size_shift, len) -> result on TOS after ret
     stbp 8             ; store result at [bp+8] and pop
     ret

uint:                   ; Builtin constructor: uint(size) -> array[] of uint
     ldbp 8             ; load len (size of array) from [bp+8] and push to stack
     push 1             ; push size_shift to stack
     call alloc         ; alloc(size_shift, len) -> result on TOS after ret
     stbp 8             ; store result at [bp+8] and pop
     ret

long:                   ; Builtin constructor: long(size) -> array[] of long
     ldbp 8             ; load len (size of array) from [bp+8] and push to stack
     push 2             ; push size_shift to stack
     call alloc         ; alloc(size_shift, len) -> result on TOS after ret
     stbp 8             ; store result at [bp+8] and pop
     ret

ulong:                  ; Builtin constructor: ulong(size) -> array[] of ulong
     ldbp 8             ; load len (size of array) from [bp+8] and push to stack
     push 2             ; push size_shift to stack
     call alloc         ; alloc(size_shift, len) -> result on TOS after ret
     stbp 8             ; store result at [bp+8] and pop
     ret

char:                   ; Builtin constructor: char(size) -> array[] of char
     ldbp 8             ; load len (size of array) from [bp+8] and push to stack
     push 0             ; push size_shift to stack
     call alloc         ; alloc(size_shift, len) -> result on TOS after ret
     stbp 8             ; store result at [bp+8] and pop
     ret

bool_to_byte:    ; Builtin function: bool_to_byte(b: bool) -> byte
    ret          ; value already at [bp+8], no conversion needed

byte_to_bool:    ; Builtin function: byte_to_bool(b: byte) -> bool
    ret          ; value already at [bp+8], no conversion needed

byte_to_int:     ; Builtin function: byte_to_int(b: byte) -> int
    ret          ; value already at [bp+8], already zero-extended to 32 bits

int_to_byte:     ; Builtin function: int_to_byte(i: int) -> byte
    ldbp 8       ; load int value from [bp+8]
    push 0xFF    ; mask for byte (8 bits)
    band         ; apply mask
    stbp 8       ; store result at [bp+8]
    ret

int_to_uint:     ; Builtin function: int_to_uint(i: int) -> uint
    ret          ; value already at [bp+8], no conversion needed

uint_to_int:     ; Builtin function: uint_to_int(u: uint) -> int
    ret          ; value already at [bp+8], no conversion needed

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

long_to_int:     ; Builtin function: long_to_int(l: long) -> int
    ldbp 8       ; load long value from [bp+8]
    push 0xFFFF  ; mask for int (16 bits)
    band         ; apply mask to get lower 16 bits
    stbp 8       ; store result at [bp+8]
    ret

long_to_ulong:   ; Builtin function: long_to_ulong(l: long) -> ulong
    ret          ; value already at [bp+8], no conversion needed

ulong_to_long:   ; Builtin function: ulong_to_long(u: ulong) -> long
    ret          ; value already at [bp+8], no conversion needed

helper_value:
    push 0    ; helper_value
.id0:
    jmp .id2
.id1:
    jmp .out
.id2:
    push 20 ; dec = 20
    stbp -4
    jmp .id1
.out:
    ldbp -4  ; load return value from helper_value
    stbp 8  ; store return value at bp + 8
    ret

all_operations_demo:
    push 0    ; all_operations_demo
.id0:
    jmp .id2
.id1:
    jmp .out
.id2:
    push 0  ; for return value
    call helper_value
    push 2 ; dec = 2
    mul
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 0  ; for return value
    call helper_value
    push 2 ; dec = 2
    div
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    add
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 0  ; for return value
    call helper_value
    push 10 ; dec = 10
    mod
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 0  ; for return value
    call helper_value
    push 10 ; dec = 10
    mod
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    sub
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    add
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 0  ; for return value
    call helper_value
    push 1 ; dec = 1
    shl
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 0  ; for return value
    call helper_value
    push 1 ; dec = 1
    shl
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    sub
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    add
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 0  ; for return value
    call helper_value
    push 1 ; dec = 1
    shr
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 0  ; for return value
    call helper_value
    push 1 ; dec = 1
    shr
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    sub
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    add
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 0  ; for return value
    call helper_value
    push 0 ; dec = 0
    bor
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 15 ; dec = 15
    band
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 0  ; for return value
    call helper_value
    push 15 ; dec = 15
    band
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    sub
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    add
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 0  ; for return value
    call helper_value
    push 0  ; for return value
    call helper_value
    bxor
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 1 ; dec = 1
    bor
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 0 ; dec = 0
    bnot_u
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 1 ; dec = 1
    band
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    add
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    push 2 ; dec = 2
    add
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    add
    push 65535  ; mask for int
    band        ; apply type mask
    push 0x8000  ; sign bit mask for int
    bxor         ; flip sign bit
    push 0x8000  ; prepare for subtraction
    sub           ; sign extend 16->32
    stbp -4
    jmp .id1
.out:
    ldbp -4  ; load return value from all_operations_demo
    stbp 8  ; store return value at bp + 8
    ret

main:
.id0:
    jmp .id2
.id1:
    jmp .out
.id2:
    push 0  ; for return value
    call all_operations_demo
    call int_to_byte
    call send_byte
    drop
    jmp .id1
.out:
    ret
