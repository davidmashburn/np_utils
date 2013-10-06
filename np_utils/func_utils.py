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

def compose(*functions):
    '''A compose function for python, i.e.:
       
       compose(f1,f2,f3)(x) <--> f1(f2(f3(x)))
       
       Works best with single-argument functions, but the last function
       can take any kind of arguments. ( No arrows yet ;-) )'''
    if len(functions)==1:
        return functions[0]
    else:
        return lambda *x,**kwds: functions[0](compose(*functions[1:])(*x,**kwds))

def fork(f,*functionList):
    '''A generalized fork function for python.
       Returns a new function that has the property:
         fork(a,b,c,d,...)(x,y,...) <--> a(b(x,y,...),c(x,y,...),d(x,y,...),...)
       The generated function can be read naturally as:
         "Do a to the results of b,c,d,... applied separately to something"
       
       An example, (assuming you have imported "div" from the operator module):
       >>> mean = fork(div,sum,len)
       
       which reads: "Divide the sum of something by its length"
       
       This generated function is equivalent to:
       >>> def mean(*args):
       >>>    return div(sum(*args),len(*args))
       
       This concept is derived from Ken Iverson's J language:
         http://www.jsoftware.com/
       Note, however, that the order of application is prefix instead of infix.
       Aka, the same mean function in J would be:
           mean =: sum div len
       
       In the simplest case, fork(a,b) <--> compose(a,b)'''
    return lambda *args,**kwds: f(*[i(*args,**kwds) for i in functionList])

def constf(value):
    '''Make a function that takes anything but always returns the value given here.
       Example:
           f=constf(10)
           f() -> 10
           f(14) -> 10
           f(1,5,a=6) -> 10'''
    return lambda *args,**kwds: value
