"""Exact entropy-reduction ordering on small rational anchor tables, no logs.

If M clears all joint-mass denominators, exp(-M H(H|Y)) equals a
product of rational posterior probabilities raised to integer powers. Comparing
those exact products orders conditional entropy in reverse. This intentionally
bounded helper is used only on predeclared small anchors, not arbitrary tables.
"""
from fractions import Fraction
from math import lcm
from .common import qstr, require


def certificate(inp):
    prior=list(map(Fraction,inp['prior']))
    distributions=[]
    multiplier=1
    for row in inp['p_one']:
        p=list(map(Fraction,row))
        outcomes=[]
        for b in (0,1):
            joint=[w*(v if b else 1-v) for w,v in zip(prior,p)]
            mass=sum(joint)
            for x in joint:multiplier=lcm(multiplier,x.denominator)
            outcomes.append((joint,mass))
        distributions.append(outcomes)
    require(multiplier<=1000,'entropy anchor denominator bound')
    products=[]
    for outcomes in distributions:
        score=Fraction(1)
        for joint,mass in outcomes:
            if not mass:continue
            for x in joint:
                if x:score *= (x/mass)**int(x*multiplier)
        products.append(score)
    best=max(products)
    return {'denominator_multiplier':multiplier,'exp_negative_scaled_conditional_entropy':[qstr(x) for x in products],'max_information_tests':[i for i,x in enumerate(products) if x==best]}
