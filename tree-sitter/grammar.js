/**
 * @file Парсер для 1 варианта
 * @license MIT
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

const PREC = {
  PAREN_DECLARATOR: -10,
  ASSIGNMENT: -2,
  CONDITIONAL: -1,
  DEFAULT: 0,
  LOGICAL_OR: 1,
  LOGICAL_AND: 2,
  INCLUSIVE_OR: 3,
  EXCLUSIVE_OR: 4,
  BITWISE_AND: 5,
  EQUAL: 6,
  RELATIONAL: 7,
  OFFSETOF: 8,
  SHIFT: 9,
  ADD: 10,
  MULTIPLY: 11,
  CAST: 12,
  SIZEOF: 13,
  UNARY: 14,
  CALL: 15,
  FIELD: 16,
  SUBSCRIPT: 17,
};

module.exports = grammar({
  name: "var1",

  word: $ => $.identifier,
  conflicts: $ => [
    [$.typeRef, $.expression],
],

  rules: {
    // source: sourceItem*;
    source: $ => repeat(field('sourceItem', $.sourceItem)),

    // sourceItem: funcDef
    sourceItem: $ => field('funcDef', $.funcDef),

    // funcDef: funcSignature (statement.block|';');
    funcDef: $ => seq(
      field('funcSignature', $.funcSignature),
      choice(
        field('block', $.block_content),
        ';'
      )
    ),

    // funcSignature: typeRef? identifier '(' list<argDef> ')'
    funcSignature: $ => seq(
      optional(field('returnType', $.typeRef)),
      field('name', $.identifier),
      '(',
      optional(field('params', $.list_argDef)),
      ')'
    ),

    list_argDef: $ => seq(
      field('param', $.argDef),
      repeat(seq(',', field('param', $.argDef)))
    ),

    // argDef: typeRef? identifier;
    argDef: $ => seq(
      optional(field('paramType', $.typeRef)),
      field('paramName', $.identifier)
    ),

    // typeRef (Variant 1):
    // builtin | custom(identifier) | array: typeRef '[' (',')* ']'
    // Implemented without left-recursion: base + repeat(suffix)
    typeRef: $ => seq(
      field('base', choice(
        alias(choice(
          'bool','byte','int','uint','long','ulong','char','string','void'
        ), 'builtin'),
        field('custom', $.identifier)
      )),
      repeat(field('arrSuffix', $.array_suffix))
    ),

    array_suffix: $ => seq('[', repeat(','), ']'),

    // statement variants
    statement: $ => choice(
      field('var', $.var_statement),
      field('if', $.if_statement),
      field('block', $.block_content),
      field('while', $.while_content),
      field('do', $.do_content),
      field('break', $.break_content),
      field('expression', $.expression_content),
    ),

    // var: typeRef list<identifier ('=' expr)?> ';'
    var_statement: $ => seq(
      field('typeRef', $.typeRef),
      field('list_varDef', $.list_varDef),
      ';'
    ),

    list_varDef: $ => seq(
      field('varDef', $.varDef),
      repeat(seq(',', field('varDef', $.varDef)))
    ),

    varDef: $ => seq(
      field('identifier', $.identifier),
      optional(seq(
        '=',
        field('expr', $.expression)
      ))
    ),

    // if: 'if' '(' expr ')' statement ('else' statement)?;
    if_statement: $ => prec.right(seq(
      'if',
      '(',
      field('expr', $.expression),
      ')',
      field('statement', $.statement),
      optional(seq('else', field('statement', $.statement))),
    )),

    // block: '{' statement* '}'
    block_content: $ => seq(
      '{',
      repeat(field('statement', $.statement)),
      '}'
    ),

    // while: 'while' '(' expr ')' statement;
    while_content: $ => seq(
      'while',
      '(',
      field('expr', $.expression),
      ')',
      field('statement', $.statement),
    ),

    // do: 'do' block 'while' '(' expr ')' ';';
    do_content: $ => seq(
      'do',
      field('block', $.block_content),
      'while',
      '(',
      field('expr', $.expression),
      ')',
      ';'
    ),

    break_content: $ => seq('break', ';'),

    expression_content: $ => field('expr', seq($.expression, ';')),

    // expr variants
    expression: $ => choice(
      field('unary', $.unary_expression),
      field('binary', $.binary_expression),
      field('braces', $.braces_expression),
      field('call', $.call_expression),
      field('indexer', $.indexer),
      field('place', $.identifier),
      field('literal', $.literal),
    ),

    braces_expression: $ => seq('(', field('expr', $.expression), ')'),

    literal: $ => choice($.bool, $.str, $.char, $.hex, $.bits, $.dec),

    indexer: $ => prec.left(PREC.SUBSCRIPT, seq(
      field('expr', $.expression),
      seq('[', optional(field('listExpr', $.list_expr)), ']')
    )),

    call_expression: $ => prec.left(PREC.CALL, seq(
      field('expr', $.expression),
      seq('(', optional(field('listExpr', $.list_expr)), ')')
    )),

    list_expr: $ => seq(
      field('expr', $.expression),
      repeat(seq(',', field('expr', $.expression)))
    ),

    unary_expression: $ => prec.left(PREC.UNARY, seq(
      field('unOp', alias(choice('!', '~'), $.un_op)),
      field('expr', $.expression),
    )),

    // IMPORTANT: Variant1 assignment uses '='
    // To avoid conflict, use '==' for equality (common choice)
    binary_expression: $ => {
      const table = [
        ['=',  PREC.ASSIGNMENT, 'right'],  // assignment
        ['+',  PREC.ADD,        'left'],
        ['-',  PREC.ADD,        'left'],
        ['*',  PREC.MULTIPLY,   'left'],
        ['/',  PREC.MULTIPLY,   'left'],
        ['%',  PREC.MULTIPLY,   'left'],
        ['||', PREC.LOGICAL_OR, 'left'],
        ['&&', PREC.LOGICAL_AND,'left'],
        ['|',  PREC.INCLUSIVE_OR,'left'],
        ['^',  PREC.EXCLUSIVE_OR,'left'],
        ['&',  PREC.BITWISE_AND,'left'],
        ['==', PREC.EQUAL,      'left'],
        ['!=', PREC.EQUAL,      'left'],
        ['>',  PREC.RELATIONAL, 'left'],
        ['>=', PREC.RELATIONAL, 'left'],
        ['<=', PREC.RELATIONAL, 'left'],
        ['<',  PREC.RELATIONAL, 'left'],
        ['<<', PREC.SHIFT,      'left'],
        ['>>', PREC.SHIFT,      'left'],
      ];

      return choice(...table.map(([operator, precedence, associativity]) => {
        const precFunc = associativity === 'right' ? prec.right : prec.left;
        return precFunc(precedence, seq(
          field('expr', $.expression),
          field('binOp', alias(token(operator), $.bin_op)),
          field('expr', $.expression),
        ));
      }));
    },

    // tokens
    identifier: _ => /[a-zA-Z_][a-zA-Z_0-9]*/,
    str: _ => /\"[^\"\\]*(?:\\.[^\"\\]*)*\"/,
    char: _ => /\'[^\']\'/,
    hex: _ => /0[xX][0-9A-Fa-f]+/,
    bits: _ => /0[bB][01]+/,
    dec: _ => /[0-9]+/,

    _true: _ => token('true'),
    _false: _ => token('false'),
    bool: $ => choice($._true, $._false),
  }
});
