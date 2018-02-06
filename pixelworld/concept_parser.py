"""Parse logical expressions (concepts) into an internal representation"""

from __future__ import print_function
from __future__ import absolute_import

from pprint import pprint
import re


def tokenize_concept(concept, debug=False):
    """
    Tokenize concept notation. This splits a string into a sequence of tokens, 
    discarding whitespace between tokens. BNF representation of tokens in terms
    of input characters:
        input ::= ([white] token)* [white]
        token ::= '?' | '(' | ')' | ',' | '~'  | '&' | ident
        ident ::= '[a-zA-Z0-9_\-]+'
        white ::= '\s+'
    """
    token_regexp = re.compile(r"[&?~,()]|([a-zA-Z0-9_\-])+")
    ident_regexp = re.compile(r"([a-zA-Z0-9_\-])+")
    whitespace_regexp = re.compile(r"\s+")
    pos = 0
    tokens = []
    is_idents = []
    token_spans = []
    while pos < len(concept):
        m = token_regexp.match(concept, pos=pos)
        if m:
            token = concept[m.start():m.end()]
            tokens.append(token)
            token_spans.append(m.span())
            is_idents.append(ident_regexp.match(token) is not None)
            pos = m.end()
        else:
            m = whitespace_regexp.match(concept, pos=pos)
            if m:
                pos = m.end()
            else:
                raise Exception("Invalid character %s at pos %s content '%s' of concept %s" % (concept[pos], pos, concept[pos:], concept))

    if debug:
        print("tokens:", [("IDENT:" if is_ident else "") + token
                            for token, is_ident in zip(tokens, is_idents)])

    return tokens, is_idents, token_spans


def parse_tokens(concept, tokens, is_idents, token_spans, debug=False, settings=None):
    """
    Parse tokenized concept notation. BNF representation of concepts in terms
    of input tokens:
        concept          ::= existential_list not_clause_list

        existential_list ::= existential*
        existential      ::= "?" term

        not_clause_list  ::= not_clause ('&' not_clause)*
        not_clause       ::= '~' '(' clause_list ')' | '~' clause | clause
        clause_list      ::= clause ('&' clause)*
        clause           ::= relation '(' term (',' term)* ')'
        
        relation         ::= ident
        term             ::= ident

    Every identifier in a clause's argument list must occur in an existential.

    Returns terms, positive_clauses, negative_clause_lists where:
        terms: [str]
        positive_clauses: [clause]
        negative_clause_lists: [[clause]]
        clause: (relation, arguments)
        relation: str
        arguments: (str)
    """
    if settings is None:
        settings = {}

    N = len(tokens)

    def parse_error(pos, message):
        if pos >= N:
            pos -= 1
            assert pos < len(tokens), "Broken parse_error!"
            raise Exception("Parse error: %s token after index %s; chars %s-%s contents %s following %s" %
                    (message, pos, token_spans[pos][0], token_spans[pos][1], 
                        tokens[pos], concept[token_spans[pos][1]:]))
        else:
            raise Exception("Parse error: %s token at index %s; chars %s-%s contents %s following %s" %
                (message, pos, token_spans[pos][0], token_spans[pos][1], 
                    tokens[pos], concept[token_spans[pos][1]:]))

    # existential_list ::= existential*
    # existential      ::= "?" term
    def parse_existential_list(pos):
        terms = []
        while pos < N:
            if tokens[pos] != '?':
                # Do not advance pos
                if debug:
                    print("parse_existential_list:", terms)
                return pos, terms
            pos += 1
            if pos >= N or not is_idents[pos]:
                parse_error(pos, "Expecting identifier for existential")
            terms.append(tokens[pos])
            pos += 1
        if debug:
            print("parse_existential_list:", terms)
        return pos, terms

    # clause ::= relation '(' term (',' term)* ')'
    def parse_clause(pos):    
        if not is_idents[pos]:
            parse_error(pos, "Expecting relation identifier")
        relation = tokens[pos]
        pos += 1
        if pos >= N or tokens[pos] != '(':
            parse_error(pos, "Expecting open parens")
        pos += 1
        arguments = []
        while pos < N:
            if not is_idents[pos]:
                parse_error(pos, "Expecting argument identifier")
            arguments.append(tokens[pos])
            pos += 1
            if pos >= N: 
                break
            if tokens[pos] == ')':
                pos += 1
                if debug:
                    print("parse_clause:", (relation, tuple(arguments)))
                return pos, (relation, tuple(arguments))
            elif tokens[pos] == ',':
                pos += 1
            else:
                parse_error(pos, "Expecting close parens or comma")
        parse_error(pos, "Run-on argument list: expecting close parens")

    # clause_list      ::= clause ('&' clause)*
    def parse_clause_list_and_close_parens(pos):
        clauses = []
        while pos < N:
            pos, clause = parse_clause(pos)            
            clauses.append(clause)
            if tokens[pos] == ')':
                pos += 1
                if debug:
                    print("parse_clause_list_and_close_parens:", clauses)
                return pos, clauses
            elif tokens[pos] == '&':
                pos += 1
            else:
                parse_error(pos, "Expecting either close parens or &")
        parse_error(pos, "Waiting on close parens to end the clause list")

    # not_clause_list  ::= not_clause ('&' not_clause)*
    # not_clause       ::= '~' '(' clause_list ')' | '~' clause | clause
    def parse_not_clause_list(pos):
        positive_clauses = []
        negative_clause_lists = []
        while pos < N:
            if tokens[pos] == '~':
                pos += 1
                if pos >= N:
                    parse_error(pos, "Expecting open paren or clause")
                elif tokens[pos] == '(':
                    pos += 1
                    pos, clause_list = parse_clause_list_and_close_parens(pos)
                    negative_clause_lists.append(clause_list)
                elif is_idents[pos]:
                    pos, clause = parse_clause(pos)
                    negative_clause_lists.append([clause])
            elif is_idents[pos]:
                pos, clause = parse_clause(pos)
                positive_clauses.append(clause)
            else:
                parse_error(pos, "Expecting ident or ~")
            if pos >= N:
                break
            elif tokens[pos] == '&':
                pos +=1

        terms = set()
        for clause_list in [positive_clauses] + negative_clause_lists:
            for relation, arguments in clause_list:
                terms.update(arguments)

        if debug:
            print("parse_not_clause_list:")
            print("   terms:", terms)
            print("   positive_clauses:", positive_clauses)
            print("   negative_clause_lists:", negative_clause_lists)
        return pos, terms, positive_clauses, negative_clause_lists


    # Parse concept
    pos = 0
    pos, quantified_terms = parse_existential_list(pos)
    pos, used_terms, positive_clauses, negative_clause_lists = parse_not_clause_list(pos)
    assert pos == N, "Didn't parse all tokens!"

    # Check all terms accounted for 
    if settings.get('add_self', True):
        quantified_terms += ['self']
    if not set(used_terms) <= set(quantified_terms):
        raise Exception("Terms used but not quantified over: %s" %
            (set(used_terms) - set(quantified_terms),))

    return quantified_terms, positive_clauses, negative_clause_lists


def parse_concept(concept, settings=None, debug=False):
    """
    Returns a tuple of terms, positive_clauses, negative_clause_lists, and meta.

    concept can be of two forms:
        1. A string giving the logical expression
        2. A dictionary with keys:
            'logic': logical expression
            'name': optional name for this concept
            'desc': optional name for this concept

    Example:
        "?x ?y ?z adjacent(x,y) & red(y) & ~(red(x) & adjacent(x, self)) & ~red(z)"
    returns the tuple (
        ['x', 'y', 'z', self'],
        [('adjacent', ('x', 'y')), ('red', ('y',)),
        [[('red', ('x',)), ('adjacent', ('x', 'self'))], 
         [('red', ('z',))]
    )
 
    See tokenize_concept and parse_concept for further details.
    """
    if settings is None:
        settings = {}
        
    if debug:
        print("parse_concept:", concept)
    if isinstance(concept, basestring):
        logic = concept
        meta = {'logic': concept}
    else:
        logic = concept['logic']
        meta = dict(concept)
    tokens, is_idents, token_spans = tokenize_concept(logic, debug)
    quantified_terms, positive_clauses, negative_clause_lists =\
        parse_tokens(concept, tokens, is_idents, token_spans, debug, settings)

    return quantified_terms, positive_clauses, negative_clause_lists, meta
