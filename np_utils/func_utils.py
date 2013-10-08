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

def identity(x):
    '''Identity function; just returns the argument'''
    return x

def packargs(*x):
    '''Returns an unpacked list of arguments as a tuple instead
       The opposite of the * operator'''
    return x

def mapf(f):
    '''Just the functional form of map:
       mapf(f)(x) <--> map(f,x)'''
    return lambda x: map(f,x)

def convertToSingleArgFun(f):
    '''Take a function that takes multiple arguments and create a function
       that takes a single argument with multiple values'''
    return lambda x,**kwds: f(*x,**kwds)

def convertToMultiArgFun(f):
    '''Take a function that takes a single argument with multiple values
       and create a function that takes a multiple arguments'''
    return lambda *args,**kwds: f(args,**kwds)

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
       
       compose(f,g)(x,y,z,...) <--> f(g(x),g(y),g(z),...)
       
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

def fork(f,*functionOrValuesList):
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
           fork(a,b) <--> a(b)'''
    return lambda *args,**kwds: f(*[ ( i(*args,**kwds)
                                       if hasattr(i,'__call__') else
                                       i )
                                    for i in functionOrValuesList])

def fork_strict(f,*functionList, **forkKwds):
    '''The same as fork, but restricted to functions only.
       
       Here is the documentation for fork:
       '''+fork.__doc__
    return lambda *args,**kwds: f(*[i(*args,**kwds) for i in functionList])
