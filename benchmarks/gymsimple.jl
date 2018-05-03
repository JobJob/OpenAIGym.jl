using PyCall, OpenAIGym
using BenchmarkTools
const env = GymEnv("CartPole-v0");
# @pyimport roboschool
# const env = GymEnv("RoboschoolHalfCheetah-v1");
# const env = cartenv = GymEnv("CartPole-v0")
# const env = GymEnv("Pong-v4");
# function gen_data(env::GymEnv, N::Int)
function gen_data(env, N::Int)
    steps = 0
    all_actions = collect(env.actions.items)::Vector{Int}

    # local s::PyArray{Float64,1}; r::Float64 = 0.0; isdone::Bool=false; _infoo_::PyObject = PyNULL()
    isdone = false
    i = 0
    while true
        i += 1
        # pycall(envresetfn, PyObject)
        @timeit2 to "resetfn" resetfn()
        while true
        # while steps < 20
            steps += 1
            # state, reward, isdone, info = regular_step(env, rand(env.actions))
            # state, reward, isdone, info = pyvec_step(env, rand(env.actions))
            # state, reward, isdone, info = nocopy_step(env, rand(env.actions))
            # state, reward, isdone, info = stepfn_step(env, rand(actions))
            @timeit2 to "choose a" a = rand(all_actions)
            @timeit2 to "stepfn_step" s, r = step!(env, a)
            # s, r, isdone, _infoo_ = stepfn_step(p0)
            isdone && break
        end
        i >= N && break
    end
    steps, s
    # steps, 1, 2
end

# steps, state = gen_data(env, 1);
# dsteps, state = gen_data(2);
