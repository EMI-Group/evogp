#pragma once

#include <stdint.h>

constexpr auto MAX_STACK = 1024, MAX_FULL_DEPTH = 10;

constexpr auto DELTA = 1E-3f, LOG_NEG = -1.0f;

typedef enum NodeType {
	VAR = 0,   // variable
	CONST = 1, // constant
	UFUNC = 2, // unary function
	BFUNC = 3,  // binary function
	TFUNC = 4, // ternary function
	TYPE_MASK = 0x7F, // node type mask
	OUT_NODE = 1 << 7, // out node flag
	UFUNC_OUT = UFUNC + OUT_NODE, // unary function, output node
	BFUNC_OUT = BFUNC + OUT_NODE,  // binary function, output node
	TFUNC_OUT = TFUNC + OUT_NODE,  // ternary function, output node
} ntype_t;

typedef enum Function {
	IF,  // arity: 3, if (a > 0) { return b } return c
	ADD, // arity: 2, return a + b
	SUB, // arity: 2, return a - b
	MUL, // arity: 2, return a * b
	DIV, // arity: 2, if (b == 0) { b = DELTA } return a / b
	MAX, // arity: 2, if (a > b) { return a } return b
	MIN, // arity: 2, if (a < b) { return a } return b
	LT,  // arity: 2, if (a < b) { return 1 } return -1
	GT,  // arity: 2, if (a > b) { return 1 } return -1
	LE,  // arity: 2, if (a <= b) { return 1 } return -1
	GE,  // arity: 2, if (a >= b) { return 1 } return -1
	SIN, // arity: 1, return sin a
	COS, // arity: 1, return cos a
	SINH,// arity: 1, return sinh a
	COSH,// arity: 1, return cosh a
	LOG, // arity: 1, if (a <= 1/e) { return LOG_NEG } return log a
	EXP, // arity: 1, return exp a
	INV, // arity: 1, if (a == 0) { a = DELTA } return 1 / a
	NEG, // arity: 1, return -a
	POW2,// arity: 1, return a * a
	POW3,// arity: 1, return a * a * a
	SQRT,// arity: 1, return if (a < 0) { return LOG_NEG } return sqrt(a)

	END  // not used, the ending notation
} func_t;