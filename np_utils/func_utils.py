'''A set of functional utilities for python:

Simple functions:
    identity -> identity function
    packargs -> turn a list of arguments into a tuple

Functions that generate functions:
    mapf -> make a function from the map of a function
    compose -> (multiple) function composition
    fork -> J language style hierarchical function combination
    constf -> make a function that always returns the same value
'''
from builtins import map

from functools import wraps
import inspect

from future.utils import viewitems, lmap

def identity(x):
    '''Identity function; just returns the argument'''
    return x

def packargs(*x):
    '''Returns an unpacked list of arguments as a tuple instead
       The opposite of the * operator'''
    return x

def mapf(f):
    '''Just the functional form of map:
       mapf(f)(x) <--> map(f,x)
       This now returns an iterator to match Python 3's map'''
    return lambda *args: map(f, *args)

def lmapf(f):
    '''Just the functional form of lmap:
       lmapf(f)(x) <--> lmap(f,x)'''
    return lambda *args: lmap(f, *args)

def mapd(f, d, *extra_args, **kwds):
    '''Map a function over dictionary values.
       Keys remain the same in the output'''
    return {k: f(v, *extra_args, **kwds)
            for k, v in viewitems(d)}

def map_in_place(f, l, *extra_args, **kwds):
    '''Mutating version of map that can only take a single argument
       Returns modified version of original list
       All optional arguments are passed to f'''
    for i, p in enumerate(l):
        l[i] = f(p, *extra_args, **kwds)

    return l

def mapd_in_place(f, d, *extra_args, **kwds):
    '''Mutating version of mapd
       Returns modified version of original dict
       All optional arguments are passed to f'''
    for k, v in viewitems(d):
        d[k] = f(v, *extra_args, **kwds)

    return d
    
def kwdPop(kwds,key,defaultValue):
    '''This is obsolete: kwds.pop(key, defaultValue) does the same thing.
       The technique is still really useful, though!
       
       If a dictionary has a key, pop the value and return it,
       otherwise return defaultValue.
       
       Allows treating missing kwd arguments as having a default value.
       Lets functions that modify **kwds arguments before passing
       them to another function to still have easy default values.
       
       If you use this, always document the behavior in the docstring.
       
       Examples:
       
       def f(*args,**kwds):
           """Has g's default parameters plus two additional kwd args:
              offset=0
              scale=1"""
           offset = kwdPop(kwds,'offset',0)
           scale = kwdPop(kwds,'scale',1)
           return scale*g(*args,**kwds) + offset
       
       def GetSwallowAirSpeed(*args,**kwds):
           """Calls AfricanAirSpeed or EuropeanAirSpeed with *args and **kwds
              Has an additional keyword argument:
                 swallowType = 'African'
              """
           swallowType = kwdPop(kwds,'swallowType','African')
           if swallowType=='African':
               return AfricanAirSpeed(*args,**kwds)
           else:
               return EuropeanAirSpeed(*args,**kwds)
       '''
    return kwds.pop(key, defaultValue)
    #( kwds.pop(key) if key in kwds else defaultValue )

def add_kwd_defaults(old_kwds, **kwds):
    '''Hack to allow defaults to be set inside the function instead of
       in the function header
       
       Example usage where g gets called inside f with the same args:
       def f(a=5, b=6, c=7, flag=-1, verbose=True):
           return g(a=a, b=b, a=5, b=b, c=c, flag=flag, verbose=verbose)
       
       using add_kwd_defaults becomes:
       
       def f(**kwds)
           kwds = add_kwd_defaults(kwds, a=5, b=6, c=7, flag=-1, verbose=True)
           return g(**kwds)
       
       Most useful in "middle-man" functions that have lots of kwds
       and need to change a small number of defaults'''
    kwds.update(old_kwds)
    return kwds

def docAppend(newFun,oldFun):
    '''Append oldFun's docstring to the end of newFun's docstring
       Useful for quick documentation of functional modifications'''
    newFun.__doc__ = '\n'.join([ (newFun.__doc__ if newFun.__doc__ else ''),
                                 '\n\n'+oldFun.__name__+':',
                                 (oldFun.__doc__ if oldFun.__doc__ else ''), ])

def convertToSingleArgFun(f):
    '''Take a function that takes multiple arguments and create a function
       that takes a single argument with multiple values'''
    return lambda x,**kwds: f(*x,**kwds)

def convertToMultiArgFun(f):
    '''Take a function that takes a single argument with multiple values
       and create a function that takes a multiple arguments'''
    return lambda *args,**kwds: f(args,**kwds)

def tryOrNone(f,*args,**kwds):
    '''A simplified functional form of a try/except;
       returns None if there is an exception.
       For a more generic verions, use "tryOrReturnException"'''
    try:
        return f(*args,**kwds)
    except:
        return None

def tryOrReturnException(f,exception=Exception,*args,**kwds):
    '''A functional form of a try/except;
       returns None if there is an exception.'''
    try:
        return f(*args,**kwds)
    except Exception as e:
        return e

def compose(*functions):
    '''A compose function for python, i.e.:
       
       compose(f1,f2,f3)(x) <--> f1(f2(f3(x)))
       
       Works best with single-argument functions, but the last function
       can take any kind of arguments, i.e.:
       
       compose(f1,f2,f3)(x,y,z) <--> f1(f2(f3(x,y,z)))'''
    if len(functions)==1:
        return functions[0]
    else:
        return lambda *x,**kwds: functions[0](compose(*functions[1:])(*x,**kwds))

def composem(f,g):
    '''A mapping compose function for python, i.e.:
       
       composem(f,g)(x,y,z,...) <--> f(g(x),g(y),g(z),...)
       
       keyword arguments go to the g function (same as compose)'''
    return lambda *args,**kwds: f(*[g(i,**kwds) for i in args])

def constf(value):
    '''Make a function that takes anything but always returns the value given here.
       Example:
           f=constf(10)
           f() -> 10
           f(14) -> 10
           f(1,5,a=6) -> 10'''
    return lambda *args,**kwds: value

def g_inv_f_g(f, g, g_inv=None):
    """Transform function f using the function g and its inverse g_inv:
    So, newf === g_inv(f(g))
    or more precisely:
    newf(*args , *kwds) === g_inv(f(*map(g, args), **kwds))
    
    By default, g is assumed to be its own inverse (g_inv=None)
    g is applied to all arguments of the new function before passing them to f
    
    This function intentionally does not handle doc strings; they should
    be handled manually instead.
    """
    g_inv = g if g_inv is None else g_inv
    def newf(*args, **kwds):
        new_args = map(g, args)
        return g_inv(f(*new_args, **kwds))
    
    return newf

def multicall(funs, *args, **kwds):
    '''Call multiple functions with the same arguments
       Essentially like map, but swapping the roles of functions and arguments
       The defalt option use_strict=False allows non-callable objects
       to be passed thru uncalled and without throwing errors'''
    use_strict = kwds.pop('use_strict', False)
    if use_strict:
        return [f(*args, **kwds) for f in funs]
    else:
        return [f(*args, **kwds) if hasattr(f, '__call__') else f
                for f in funs]

def fork(f, *functionOrValuesList, **fkwds):
    '''A generalized fork function for python which additionally allows
           non-callable values to be passed through.
       
       Takes a function, any number of additional functions.
       Returns a new function that has the property:
         fork(a,b,c,d,...)(x,y,...) <--> a(b(x,y,...),c(x,y,...),d(x,y,...),...)
       or if an item in the list is not a function (for instance, c below):
         fork(a,b,c,d,...)(x,y,...) <--> a(b(x,y,...),c,d(x,y,...),...)
       The generated function can be read naturally as:
         "Do a to the results of b,c,d,... applied separately to something"
       
       An example, (assuming you have imported "div" from the operator module):
       >>> mean = fork(div,sum,len)
       This reads "Divide the sum of something by its length"
       
       This generated function is equivalent to:
       >>> def mean(*args):
       >>>    return div(sum(*args),len(*args))
       
       Or, using a non-callable:
       >>> mean = fork(div,sum,2)
       This reads: "Divide the sum of something by 2"
       
       This concept is derived from Ken Iverson's J language:
         http://www.jsoftware.com/
       Note, however, that the order of application is prefix instead of infix.
       Aka, the same mean function in J would be:
           mean =: sum div len
       
       In the simplest case with b as a function, fork is equivalent to "compose":
           fork(a,b) <--> compose(a,b)
       Alternatively, with b as a value, fork is equivalent to "call"
           fork(a,b) <--> a(b)
       
       fork also takes two optional kwds (the rest are passed to f):
           use_splat: whether or not to unpack the arguments when calling f
                      default: True
           use_strict: when False, forces all args to be functions
                       or else errors will occur
                       default: False'''
    use_splat = fkwds.pop('use_splat', True)
    use_strict = fkwds.pop('use_strict', False)
    
    def newf(*args, **kwds):
        kwds['use_strict'] = use_strict
        fargs = multicall(functionOrValuesList, *args, **kwds)
        fargs = fargs if use_splat else [fargs]
        return f(*fargs, **fkwds)
    return newf

def old_fork(f, *functionOrValuesList):
    return lambda *args, **kwds: (
        f(*[i(*args,**kwds) if hasattr(i,'__call__') else i
            for i in functionOrValuesList]))

def fork1(f,*functionOrValuesList, **fkwds):
    '''Same as fork, but f expects a single argument with multiple elements
      (non-splat) instead of multiple arguments (splat)
       
       Here is the documentation for fork:
       ''' + fork.__doc__
    fkwds['use_splat'] = False
    return fork(f, *functionOrValuesList, **fkwds)

def fork_strict(f, *functionList, **fkwds):
    '''The same as fork, but restricted to functions only.
       
       Here is the documentation for fork:
       '''+fork.__doc__
    fkwds['use_strict'] = True
    return fork(f, *functionList, **fkwds)

def doublewrap(f):
    '''
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    lifted from this StackOverflow answer:
    http://stackoverflow.com/questions/653368/how-to-create-a-python-decorator-that-can-be-used-either-with-or-without-paramet
    '''
    @wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0]) # use the basic decorator pattern
        else:
            # decorator arguments
            def realf(realf):
                return f(realf, *args, **kwargs) # use the nested decorator pattern
            return realf

    return new_dec

def flipargs(f):
    '''Generator that changes a function to take its arguments in reverse'''
    @wraps(f)
    def newf(*args, **kwds):
        return f(*reversed(args), **kwds)
    
    # Add a note about the args being flipped to the doc string
    newf.__doc__ = 'Arguments reversed:\n\n' + newf.__doc__
    return newf

def reflex(f):
    '''Take a function that takes two arguments and change it to take
       only one argument that is used as both arguments
       Any kwds are passed through as normal'''
    @wraps(f)
    def newf(x, **kwds):
        return f(x, x, **kwds)
    
    return newf

#############################
## inspect-based utilities ##
#############################

def get_function_arg_names_and_kwd_values(f):
    '''Use the inspect module to extract the args and kwds from function
       Returns the argument names as a list of strings and
       the keyword values as a list of objects.
       
       There will always be fewer (or euqal) keyword values than
       argument names, and keyword values always line up with the end of
       the argument names
       
       This cannot handle functions that use splatting (*args or **kwds)
       
       Example:
       >>> def f(a, b, c=5): return a + b + c
       >>> get_function_args_and_kwds(f)
       (['a', 'b', 'c'], (5,))
       >>> get_function_args_and_kwds(f, skip=1)
       (['b', 'c'], (5,))
       '''
    a = inspect.getargspec(f)
    assert a.varargs is None and a.keywords is None, \
        "Arg maties! This function doesn't handle *args or **kwds"
    return a.args, (() if a.defaults is None else a.defaults)
