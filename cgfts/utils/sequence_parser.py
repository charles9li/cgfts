__all__ = ['acrylate_sequence_to_list']


_OPERATOR_PRECEDENCE = {'+': 2,
                        '*': 3}
_OPERATORS = list(_OPERATOR_PRECEDENCE.keys())
_DELIMITERS = ['(', ')']


def acrylate_sequence_to_list(sequence):
    return evaluate_output(create_stack(tokenize_expr(sequence)))


def tokenize_expr(expr):
    token_list = expr.split()
    for char in _OPERATORS + _DELIMITERS:
        temp_list = []
        for token in token_list:
            token = [t.strip() for t in token.split(char)]
            token_split = [char] * (len(token) * 2 - 1)
            token_split[0::2] = token
            temp_list += token_split
        token_list = temp_list
    return [t for t in token_list if t]


def create_stack(token_list):
    output = []
    stack = []
    for token in token_list:
        if token in _OPERATORS:
            while len(stack) > 0 and stack[-1] != '(' and _OPERATOR_PRECEDENCE[stack[-1]] >= _OPERATOR_PRECEDENCE[token]:
                output.append(stack.pop())
            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        else:
            output.append(token)
    while len(stack) > 0:
        output.append(stack.pop())
    return output


def evaluate_output(output):
    stack = []
    for item in output:
        if item in _OPERATORS:
            op2 = stack.pop()
            op1 = stack.pop()
            if item == '+':
                stack.append(op1 + op2)
            elif item == '*':
                stack.append(op1 * op2)
        else:
            try:
                stack.append(int(item))
            except ValueError:
                stack.append([item])
    if len(stack) != 1:
        raise ValueError("Malformed expression.")
    return stack.pop()
