module OpenAIGym

using PyCall
export
    GymEnv,
    step!,
    reset!,
    finished,
    rand_action,
    render,
    PyAny


const _py_envs = Dict{String,Any}()

# --------------------------------------------------------------

"A wrapper around the Python OpenAI gym environments"
mutable struct GymEnv{T}
    name::String
    pyenv::PyObject   # the python "env" object
    pystep::PyObject  # the python env.step function
    pyreset::PyObject # the python env.reset function
    pystate::PyObject # the state array object referenced by the PyArray state.o
    pystepres::PyObject # used to make stepping the env slightly more efficient
    pytplres::PyObject  # used to make stepping the env slightly more efficient
    info::PyObject    # store it as a PyObject for speed, since often unused
    state::T
    reward::Float64
    total_reward::Float64
    actions::PyObject
    done::Bool
    function GymEnv{T}(name, pyenv, pystate, state::T) where T
        env = new{T}(name, pyenv, pyenv["step"], pyenv["reset"],
                         pystate, PyNULL(), PyNULL(), PyNULL(), state)
        reset!(env)
        env
    end
end

function GymEnv(name; stateT=PyArray)
    env = if name in ("Soccer-v0", "SoccerEmptyGoal-v0")
        Base.copy!(gym_soccer, pyimport("gym_soccer"))
        get!(_py_envs, name) do
            GymEnv(name, pygym[:make](name), stateT)
        end
    else
        GymEnv(name, pygym[:make](name), stateT)
    end
    env
end

function GymEnv(name, pyenv, stateT)
    pystate = pycall(pyenv["reset"], PyObject)
    state = convert(stateT, pystate)
    T = typeof(state)
    GymEnv{T}(name, pyenv, pystate, state)
end


# --------------------------------------------------------------

render(env::GymEnv, args...; kwargs...) =
    pycall(env.pyenv[:render], PyAny; kwargs...)

# --------------------------------------------------------------

pyaction(a::Vector) = Any[pyaction(ai) for ai=a]
pyaction(a) = a

"""
`reset!` for PyArray state types
"""
function reset!(env::GymEnv{T}) where T <: PyArray
    setdata!(env.state, pycall!(env.pystate, env.pyreset, PyObject))
    return gymreset!(env)
end

"""
`reset!` for non PyArray state types
"""
function reset!(env::GymEnv{T}) where T
    pycall!(env.pystate, env.pyreset, PyObject)
    env.state = convert(T, env.pystate)
    return gymreset!(env)
end

function gymreset!(env::GymEnv{T}) where T
    env.reward = 0.0
    env.total_reward = 0.0
    env.actions = env.pyenv["action_space"]
    env.done = false
    return env.state
end

"""
`step!` for PyArray state
"""
function step!(env::GymEnv{T}, a) where T <: PyArray
    pyact = pyaction(a)
    pycall!(env.pystepres, env.pystep, PyObject, pyact)

    unsafe_gettpl!(env.pystate, env.pystepres, PyObject, 0)
    setdata!(env.state, env.pystate)

    return gymstep!(env)
end

"""
step! for non-PyArray state
"""
function step!(env::GymEnv{T}, a) where T
    pyact = pyaction(a)
    pycall!(env.pystepres, env.pystep, PyObject, pyact)

    unsafe_gettpl!(env.pystate, env.pystepres, PyObject, 0)
    env.state = convert(T, env.pystate)

    return gymstep!(env)
end

@inline function gymstep!(env)
    r = unsafe_gettpl!(env.pytplres, env.pystepres, Float64, 1)
    env.done = unsafe_gettpl!(env.pytplres, env.pystepres, Bool, 2)
    unsafe_gettpl!(env.info, env.pystepres, PyObject, 3)
    env.total_reward += r
    return (r, env.state)
end

finished(env::GymEnv) = env.done
finished(env::GymEnv, s′) = env.done

rand_action(env::GymEnv) = env.actions["sample"]()

# --------------------------------------------------------------

global const pygym = PyNULL()
global const pysoccer = PyNULL()

function __init__()
    # the copy! puts the gym module into `pygym`, handling python ref-counting
    Base.copy!(pygym, pyimport("gym"))
end

end # module
