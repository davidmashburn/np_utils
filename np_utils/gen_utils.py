'''Some general utilities functions:
Conversion utilities:
    intOrFloat -> converts string to either int if possible, float otherwise
    floatIntStringOrNone -> try to make an integer, then a float, then a string, otherwise None

Flow control utilities:
    minmax -> Return the tuple (min(...), max(...))
    callFunctionIfNotNone -> Take a function of 2 arguments (a and b) and return
                             None if both a and b are None,
                             a if b is None (and vice versa),
                             and f(a,b) if neither is None.
    minmaxIgnoreNone -> A poorly named function that takes 2 minima and
                        2 maixima and returns the global min and max
                        using callFunctionIfNotNone
'''
from builtins import map, int


import os
import types
import re

#############################
## File handling utilities ##
#############################

def makeifnotexists(d):
    '''If a directory does not exist, make it
       Return d (for functional use cases)'''
    if not os.path.exists(d):
        os.mkdir(d)
    return d

##########################
## Conversion utilities ##
##########################

def intOrFloat(string):
    '''Not sure if your string is formatted as an int or a float? Use intOrFloat instead!'''
    try:
        return int(string)
    except ValueError:
        return float(string)

def floatIntStringOrNone(string):
    '''An even more generic version of intOrFloat'''
    if string=='None':
        return None

    for fun in (int, float):
        try:
            return fun(string)
        except ValueError:
            pass

    return string

def islistlike(x):
    '''Test if something is an iterable but NOT as string'''
    return hasattr(x, '__iter__') and not isinstance(x, str)

######################
## String utilities ##
######################

def string_between(s, before, after):
    '''Find the string between two substrings
       Bascially rolling up the pattern:
          s.split(before)[1].split(after)[0]
       into an actual function with proper checks.
       To start from the beginning or end of the string,
       pass None as before or after'''
    if before is not None:
        t = s.split(before)
        assert len(t) > 1, '"before" argument is not in the string!'
        s = t[1]
    if after is not None:
        t = s.split(after)
        assert len(t) > 1, '"after" argument is not in the string!'
        s = t[0]
    return s

def multisplit(string, *delimiters):
    '''Split a string at any of a number of delimeters.
       With one delimeter, this is equivalent to string.split.'''
    pattern = '|'.join(map(re.escape, delimiters))
    return re.split(pattern, string)

def multireplace(text, *replpairs):
    '''Chain multiple calls of string.replace
       A "re"-based approach may be better for very long strings
       and/or many replacements:
       http://code.activestate.com/recipes/81330-single-pass-multiple-replace/'''
    for i,o in replpairs:
        text = text.replace(i,o)
    return text

def multiremove(text, *removals):
    '''Chain multiple calls of string.replace
       where the second argument is always '' '''
    for r in removals:
        text = text.replace(r,'')
    return text

############################
## Flow control utilities ##
############################

def minmax(*args,**kwds):
    '''A really simple function that makes it cleaner to get the min and max
       from an expression without duplication or creating a local variable.
       See builtins min and max for details about arguments.'''
    return min(*args,**kwds),max(*args,**kwds)

def callFunctionIfNotNone(f,a,b):
    ''''A really simple function to call a function only if both arguments
        are not None'''
    if a==None:   return b
    elif b==None: return a
    else:         return f(a,b)

def minmaxIgnoreNone(Amin,Bmin, Amax,Bmax): ## pairwiseMinMaxIgnoreNone(Amin,Bmin, Amax,Bmax):
    '''Given two minima and two maxima, calculate the global minima and maxima,
       ignoring values that are None

       TODO: This should be renamed (maybe pairwiseMinMaxIgnoreNone?) to avoid confusion with minmax
       BTW, when you do fix the name, realize that the max was computing mins instead!!!!''' #TODO
    return callFunctionIfNotNone(min,Amin,Bmin),callFunctionIfNotNone(max,Amax,Bmax)
