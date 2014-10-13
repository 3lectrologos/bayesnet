#!/usr/bin/env python
"""
Message Passing for Factor Graphs
=================================

We are using a very simple schedule. First, we initialize all messages from
variable nodes to factors to be uniform. Then, we alternate between passing
vector -> factor and factor -> vector messages.

Note: For numerical stability we use log-probabilities, thus products turn into
      sums.
"""

from __future__ import division
import argparse
from collections import defaultdict, namedtuple

from matplotlib import pyplot
import networkx
import numpy as np
from scipy.misc import logsumexp

NODE_SIZE = 9000  # The size of the drawn graph nodes.
EPS = 1e-10  # Numerical sensitivity.


Factor = namedtuple('Factor', 'variables table')
"""Represents a factor. The fields are as following:

   * ``variables``: list of variables that are in the factor
                    (the order **does** matter)
   * ``table``: for every **tuple** of values y=(x_1, ..., x_n) the variables
                can take (in the order of ``variables``), the factor will be
                evaluated at using table[y]."""


def check_probability_table(table):
    """If the table is a conditional probability distribution, checks that the
    values make sense. The last column of the table should be the variable over
    which the distribution is specified."""
    probabilities = defaultdict(list)
    for (line, value) in table.items():
        probabilities[line[:-1]].append(value)

    for distribution in probabilities.values():
        assert all(x >= 0 and x <= 1 for x in distribution)
        assert abs(sum(distribution) - 1) <= EPS


def compute_variable_messages(messages, values):
    """Compute messages variable -> factor."""
    # The factors in which the variable participates (the ones it has received
    # a message from).
    factors = tuple(messages.keys())

    output = dict()  # The outgoing messages.
    for (i, factor_i) in enumerate(factors):
        # Compute the message going to ``factor_i``.
        output[factor_i] = defaultdict(float)
        for (j, factor_j) in enumerate(factors):
            if i == j:  # Ignore the message from ``factor_i``.
                continue
            for value in values:
                # We are working with log-probabilities, so we have to sum.
                output[factor_i][value] += messages[factor_j][value]
    return output


def compute_factor_messages(messages, factor):
    """Compute messages factor -> variable."""
    output = dict()

    for (i, variable) in enumerate(factor.variables):
        # Because we are working with log-probabilities, not to run into
        # numerical issues we will first accumulate all the summands.
        output[variable] = defaultdict(list)
        for (combination, factor_value) in factor.table.items():
            if abs(factor_value) <= EPS:  # It does not contribute to the sum.
                continue
            log_factor = np.log(factor_value)
            log_prob = 0.0
            for (j, variable_j) in enumerate(factor.variables):
                if i == j:  # Ignore the message from ``variable_i``.
                    continue
                log_prob += messages[variable_j][combination[j]]

            output[variable][combination[i]].append(log_factor + log_prob)

    # Evaluate the list of values using ``logsumexp``.
    for key_1 in output:
        for key_2 in output[key_1]:
            output[key_1][key_2] = logsumexp(output[key_1][key_2])

    return output


def draw_factor_graph(factors):
    graph = networkx.Graph()
    for (factor_name, factor) in factors.items():
        for variable in factor.variables:
            graph.add_edge(factor_name, variable)
    colors = []
    for node in graph.nodes():
        if node in factors.keys():
            colors.append('gray')
        else:
            colors.append('green')
    pyplot.figure()
    networkx.draw_networkx(graph, with_labels=True, node_color=colors,
                           node_size=NODE_SIZE)
    pyplot.title('The Factor Graph')
    pyplot.show()


def compute_marginals(variables, messages, verbose=True):
    marginals = defaultdict(dict)
    for (variable, values) in sorted(variables.items()):
        if verbose:
            print(variable)
            print('-'*len(variable))
        incoming = defaultdict(float)
        for messages_value in messages[variable].values():
            for (value, logp) in messages_value.items():
                incoming[value] += logp
        normalizer = logsumexp(tuple(incoming.values()))
        for value in values:
            marginal = np.exp(incoming[value] - normalizer)
            marginals[variable][value] = marginal
            if verbose:
                print('  *  {0} : {1:.4f}'.format(value, marginal))
    if verbose:
        print('='*40)
    return marginals


def pass_messages(variables, factors, iterations, verbose=True,):
    # Messages sent from factors to variables.
    messages_variables = defaultdict(lambda: defaultdict(dict))
    # Messages that are sent from variables to factors.
    messages_factors = defaultdict(lambda: defaultdict(dict))

    # Initialize the variable -> factor messages to uniform distributions.
    for (factor_name, factor) in factors.items():
        for variable in factor.variables:
            # We work with log probabilities.
            values = variables[variable]
            probability = -np.log(len(values))
            messages_factors[factor_name][variable] = \
                dict((value, probability) for value in values)

    for _ in range(iterations):
        messages_variables.clear()
        for (factor_name, factor) in factors.items():
            output = compute_factor_messages(
                messages_factors[factor_name], factor)
            for (recipient, message) in output.items():
                messages_variables[recipient][factor_name] = message

        messages_factors.clear()
        for (variable, values) in variables.items():
            output = compute_variable_messages(
                messages_variables[variable], values)
            for (recipient, message) in output.items():
                messages_factors[recipient][variable] = message

        if verbose:
            compute_marginals(variables, messages_variables, verbose=True)

    return compute_marginals(variables, messages_variables, verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Belief Propagation on Factor Graphs')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='print the marginals after each iteration')
    parser.add_argument('--draw', dest='draw', action='store_true',
                        help='draw the factor graph')
    parser.add_argument('--iterations', dest='iterations', action='store',
                        type=int, default=10,
                        help='number of iterations, default: 10')
    parser.set_defaults(verbose=False, draw=False)
    args = parser.parse_args()

    # For each variable we specify the set of values it can take.
    variables = {
        'variable_1': (0, 1),
        'variable_2': (0, 1),
        'variable_3': (0, 1),
    }

    factors = {
        'factor_1': Factor(
            ('variable_2', 'variable_3', 'variable_1'),
            {(0, 0, 0): 0.99,
             (0, 0, 1): 0.01,
             (0, 1, 0): 0.99,
             (0, 1, 1): 0.01,
             (1, 0, 0): 0.99,
             (1, 0, 1): 0.01,
             (1, 1, 0): 0.001,
             (1, 1, 1): 0.999}
        ),
        'factor_2': Factor(
            ('variable_2',),
            {(0,): 0.001,
             (1,): 0.999}
        ),
        'factor_3': Factor(
            ('variable_3',),
            {(0,): 0.001,
             (1,): 0.999}
        ),
    }

    # Remove this check if the factors are not conditional probability tables.
    for factor in factors.values():
        check_probability_table(factor.table)
    if args.draw:
        draw_factor_graph(factors)
    pass_messages(variables, factors, args.iterations, verbose=args.verbose)
