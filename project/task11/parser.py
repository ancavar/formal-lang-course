from antlr4 import (
    ParseTreeListener,
    ParserRuleContext,
    TerminalNode,
    CommonTokenStream,
    InputStream,
    ParseTreeWalker,
)
from project.task11.GraphLangLexer import GraphLangLexer
from project.task11.GraphLangParser import GraphLangParser


class NodeCounter(ParseTreeListener):
    def __init__(self):
        super().__init__()
        self.count = 0

    def enterEveryRule(self, ctx: ParserRuleContext):
        self.count += 1


class TokenCollection(ParseTreeListener):
    def __init__(self):
        super().__init__()
        self.tokens = []

    def visitTerminal(self, node: TerminalNode):
        self.tokens.append(node.getText())


def program_to_tree(program: str) -> tuple[ParserRuleContext, bool]:
    lexer = GraphLangLexer(InputStream(program))
    stream = CommonTokenStream(lexer)
    parser = GraphLangParser(stream)

    tree = parser.prog()

    return (tree, parser.getNumberOfSyntaxErrors() == 0)


def nodes_count(tree: ParserRuleContext) -> int:
    listener = NodeCounter()
    ParseTreeWalker().walk(listener, tree)

    return listener.count


def tree_to_program(tree: ParserRuleContext) -> str:
    listener = TokenCollection()
    ParseTreeWalker().walk(listener, tree)

    return " ".join(listener.tokens)
