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
    try:
        return int(string)
    except ValueError:
        try:
            return float(string)
        except ValueError:
            return string

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

