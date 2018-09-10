
 __precompile__()

module OpenAIGym

using PyCall
using Reexport
@reexport using Reinforce
import Reinforce:
    MouseAction, MouseActionSet,
    KeyboardAction, KeyboardActionSet

export
    gym,
    GymEnv,
    render,
    test_env

const _py_envs = Dict{String,Any}()

# --------------------------------------------------------------

abstract type AbstractGymEnv <: AbstractEnvironment end

"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
mutable struct GymEnv{T} <: AbstractGymEnv
    name::String
    pyenv::PyObject   # the python "env" object
    pystep::PyObject  # the python env.step function
    pyreset::PyObject # the python env.reset function
    pystate::PyObject # the state array object referenced by the PyArray state.o
    info::PyObject    # store it as a PyObject for speed
    state::T
    reward::Float64
    total_reward::Float64
    actions::AbstractSet
    done::Bool
    function GymEnv(name, pyenv, pyarray_state)
        state_type = pyarray_state ? PyArray : PyAny
        state = pycall(pyenv["reset"], state_type) # initialise env.state
        env = new{typeof(state)}(name, pyenv, pyenv["step"], pyenv["reset"], PyNULL(), PyNULL(), state)
        reset!(env)
        env
    end
end
GymEnv(name; pyarray_state=false) = gym(name; pyarray_state=pyarray_state)

function gymreset!(env::GymEnv{T}) where T
    env.reward = 0.0
    env.total_reward = 0.0
    env.actions = actions(env, nothing)
    env.done = false
    return env.state
end

function Reinforce.reset!(env::GymEnv{T}) where T <: PyArray
    setdata!(env.state, pycall!(env.pystate, env.pyreset, PyObject))
    return gymreset!(env)
end

"""
Non PyArray state types
"""
function Reinforce.reset!(env::GymEnv{T}) where T
    pycall!(env.pystate, env.pyreset, PyObject)
    env.state = convert(T, env.pystate)
    return gymreset!(env)
end

"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
mutable struct UniverseEnv <: AbstractGymEnv
    name::String
    pyenv  # the python "env" object
    state
    reward
    total_reward
    actions::AbstractSet
    done
    info::Dict
    UniverseEnv(name,pyenv) = new(name,pyenv)
end
UniverseEnv(name) = gym(name)

function Reinforce.reset!(env::UniverseEnv)
    env.state = env.pyenv[:reset]()
    env.reward = [0.0]
    env.total_reward = 0.0
    env.actions = actions(env, nothing)
    env.done = [false]
end

function gym(name::AbstractString; pyarray_state=false)
    env =
    if name in ("Soccer-v0", "SoccerEmptyGoal-v0")
        global gym_soccer = pyimport("gym_soccer") # see https://github.com/JuliaPy/PyCall.jl/issues/541
        get!(_py_envs, name) do
            GymEnv(name, pygym[:make](name))
        end
    # elseif split(name, ".")[1] in ("flashgames", "wob")
    #     global universe
    #     @pyimport universe
    #     @pyimport universe.wrappers as wrappers
    #     if !isdefined(OpenAIGym, :vnc_event)
    #         global const vnc_event = PyCall.pywrap(PyCall.pyimport("universe.spaces.vnc_event"))
    #     end
    #     get!(_py_envs, name) do
    #         pyenv = wrappers.SafeActionSpace(pygym[:make](name))
    #         pyenv[:configure](remotes=1)  # automatically creates a local docker container
    #         # pyenv[:configure](remotes="vnc://localhost:5900+15900")
    #         o = UniverseEnv(name, pyenv)
    #         # finalizer(o,  o.pyenv[:close]())
    #         sleep(2)
    #         o
    #     end
    else
        GymEnv(name, pygym[:make](name), pyarray_state)
    end
    env
end


# --------------------------------------------------------------

render(env::AbstractGymEnv, args...; kwargs...) =
    pycall(env.pyenv[:render], PyAny; kwargs...)

# --------------------------------------------------------------


function actionset(A::PyObject)
    if haskey(A, :n)
        # choose from n actions
        DiscreteSet(0:A[:n]-1)
    elseif haskey(A, :spaces)
        # a tuple of action sets
        sets = [actionset(a) for a in A[:spaces]]
        TupleSet(sets...)
    elseif haskey(A, :high)
        # continuous interval
        IntervalSet{Vector{Float64}}(A[:low], A[:high])
        # if A[:shape] == (1,)  # for now we only support 1-length vectors
        #     IntervalSet{Float64}(A[:low][1], A[:high][1])
        # else
        #     # @show A[:shape]
        #     lo,hi = A[:low], A[:high]
        #     # error("Unsupported shape for IntervalSet: $(A[:shape])")
        #     [IntervalSet{Float64}(lo[i], hi[i]) for i=1:length(lo)]
        # end
    elseif haskey(A, :buttonmasks)
        # assumed VNC actions... keys to press, buttons to mask, and screen position
        # keyboard = DiscreteSet(A[:keys])
        keyboard = KeyboardActionSet(A[:keys])
        buttons = DiscreteSet(Int[bm for bm in A[:buttonmasks]])
        width,height = A[:screen_shape]
        mouse = MouseActionSet(width, height, buttons)
        TupleSet(keyboard, mouse)
    elseif haskey(A, :actions)
        # Hardcoded
        TupleSet(DiscreteSet(A[:actions]))
    else
        @show A
        @show keys(A)
        error("Unknown actionset type: $A")
    end
end


function Reinforce.actions(env::AbstractGymEnv, s′)
    actionset(env.pyenv["action_space"])
end

pyaction(a::Vector) = Any[pyaction(ai) for ai=a]
pyaction(a::KeyboardAction) = Any[a.key]
pyaction(a::MouseAction) = Any[vnc_event.PointerEvent(a.x, a.y, a.button)]
pyaction(a) = a

const pytplres = PyNULL()
const pystepres = PyNULL()
const pyargsptr = PyNULL()
const pyacto = PyNULL()
const pybufinfo = PyBuffer()

using PyCall: @pycheckn, @pycheckz, pydecref_, __pycall!
import PyCall: pysetarg!

function PyObject!(pyo, i::Integer)
    pydecref_(pyo.o)
    # pyo.o = @pycheckn ccall(@pysym(:PyLong_FromLongLong), PyPtr, (Clonglong,), i)
    pyo.o = ccall(@pysym(:PyLong_FromLongLong), PyPtr, (Clonglong,), i)
    pyo
end

function pysetarg!(pyargsptr, pyarg::Union{PyPtr, PyObject}, i::Int)
    pyincref(pyarg) # PyTuple_SetItem steals the reference
    # @pycheckz ccall((@pysym :PyTuple_SetItem), Cint,
    ccall((@pysym :PyTuple_SetItem), Cint,
                        (PyPtr,Int,PyPtr), pyargsptr, i-1, pyarg)
end

timestuff = false

macro timeit2(exprs...)
    if timestuff
        return :(@timeit($(esc.(exprs)...)))
    else
        return esc(exprs[end])
    end
end

# @inline function Reinforce.step!(env::GymEnv{T}, a) where T <: PyArray
@inline function Reinforce.step!(env::GymEnv{T}, a) where T <: PyArray
    pyact = pyaction(a)
    # pyargsptr.o = @pycheckn ccall((@pysym :PyTuple_New), PyPtr, (Int,), 1)
    # pyargsptr.o = ccall((@pysym :PyTuple_New), PyPtr, (Int,), 1)
    # pysetarg!(pyargsptr, PyObject!(pyacto, pyact), 1)
    # __pycall!(pystepres, pyargsptr.o, env.pystep.o, C_NULL)
    pycall!(pystepres, env.pystep, PyObject, pyact)

    unsafe_gettpl!(env.pystate, pystepres, PyObject, 0)
    setdata!(env.state, env.pystate)

    return gymstep!(env, pystepres)
    # pydecref(pyargsptr);
    # return env.state
end

"""
step! for non-PyArray state
"""
function Reinforce.step!(env::GymEnv{T}, a) where T
    pyact = pyaction(a)
    pycall!(pystepres, env.pystep, PyObject, pyact)
    unsafe_gettpl!(env.pystate, pystepres, PyObject, 0)
    env.state = convert(T, env.pystate)

    return gymstep!(env, pystepres)
end

@inline function gymstep!(env, pystepres)
    r = unsafe_gettpl!(pytplres, pystepres, Float64, 1)
    env.done = unsafe_gettpl!(pytplres, pystepres, Bool, 2)
    unsafe_gettpl!(env.info, pystepres, PyObject, 3)
    env.total_reward += r
    return (r, env.state)
end

function Reinforce.step!(env::UniverseEnv, s, a)
    info("Going to take action: $a")
    pyact = Any[pyaction(a)]
    s′, r, env.done, env.info = env.pyenv[:step](pyact)
    env.reward = r
    env.total_reward += r[1] # assuming it's an array based on `reset!`
    env.state = s′
    r, s′
end

Reinforce.finished(env::GymEnv) = env.done
Reinforce.finished(env::GymEnv, s′) = env.done
Reinforce.finished(env::UniverseEnv, s′) = all(env.done)

Reinforce.total_reward(env::GymEnv) = env.total_reward

# --------------------------------------------------------------


function test_env(name::String = "CartPole-v0")
    env = gym(name)
    for sars′ in Episode(env, RandomPolicy())
        render(env)
    end
end


global const pygym = PyNULL()

function __init__()
    # @static if is_linux()
    #     # due to a ssl library bug, I have to first load the ssl lib here
    #     condadir = Pkg.dir("Conda","deps","usr","lib")
    #     Libdl.dlopen(joinpath(condadir, "libssl.so"))
    #     Libdl.dlopen(joinpath(condadir, "python2.7", "lib-dynload", "_ssl.so"))
    # end

    pygym.o = pyimport("gym").o
end

end # module
