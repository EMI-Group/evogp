#pragma once

constexpr auto MAX_STACK = 128 * sizeof(float);

constexpr auto DELTA = 1E-3, LOG_NEG = -1.0;

typedef enum NodeType {
	VAR,   // variable
	CONST, // constant
	UFUNC, // unary function
	BFUNC,  // binary function
	//TFUNC, // trinary function
} ntype_t;

typedef enum Function {
	ADD, // arity: 2, return a + b
	SUB, // arity: 2, return a - b
	MUL, // arity: 2, return a * b
	DIV, // arity: 2, if (b == 0) { b = DELTA } return a / b
	TAN, // arity: 1, return tan a
	SIN, // arity: 1, return sin a
	COS, // arity: 1, return cos a
	LOG, // arity: 1, return log a
	EXP, // arity: 1, return exp a
	MAX, // arity: 2, if (a > b) { return a } return b
	MIN, // arity: 2, if (a < b) { return a } return b
	INV, // arity: 1, if (a == 0) { a = DELTA } return 1 / a
	NEG, // arity: 1, return -a
} func_t;