#include "Python.h"

uint64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}

PyObject* pycall1(PyObject* o, PyObject* arg1);
PyObject* pycall0(PyObject* o);
long gen_data(PyObject* stepfn, long maxsteps);
void reset(PyObject* env);

int main(int argc, char const *argv[]) {
  Py_Initialize();
  PyObject* gym = PyImport_ImportModule("gym");
  const char *envname = "PongNoFrameskip-v4";
  PyObject* envmakefn = PyObject_GetAttrString(gym, "make");
  PyObject* env = pycall1(envmakefn, PyUnicode_FromString(envname));
  // PyObject_Print(env, stdout, 0); printf("\n"); // print the name of the env

  int N = 10;
  int trials_per_sample = 1;
  uint64_t times[N];
  double step_times[N];

  reset(env);

  PyObject* stepfn = PyObject_GetAttrString(env, "step");

  unsigned long long pre;
  for (size_t i = 0; i < N; i++) {
    reset(env);
    for (size_t j = 0; j < trials_per_sample; j++) {
      unsigned long long pre = GetTimeStamp();
      // PyObject_Print(action, stdout, 0); printf("\n");
      // pycall1(stepfn, action);
      long nsteps = gen_data(stepfn, 10000);
      times[i] = GetTimeStamp() - pre;
      step_times[i] = (double)times[i] / (double)nsteps;
    }
  }

  for (size_t i = 0; i < N; i++) {
    printf("%llu   %f\n", times[i], step_times[i]);
  }

  return 0;
}

PyObject* pycall0(PyObject* o){
  PyObject* argsptr = PyTuple_New(0); //new ref
  PyObject* res = PyObject_Call(o, argsptr, NULL); //new ref
  Py_DECREF(argsptr);
  return res;
}

/*
N.b. because of the `PyTuple_SetItem`, this function "steals" the ref to arg1,
i.e. takes responsibility for calling decref on it. If you want to keep the arg1
around, be sure to incref it before calling this.
*/
PyObject* pycall1(PyObject* o, PyObject* arg1){
  PyObject* argsptr = PyTuple_New(1); //new ref
  PyTuple_SetItem(argsptr, 0, arg1);  //ref to arg1 is stolen
  PyObject* res = PyObject_Call(o, argsptr, NULL); //new ref
  Py_DECREF(argsptr);
  return res;
}

long gen_data(PyObject* stepfn, long maxsteps){
  PyObject* action;
  PyObject* res;
  PyObject* done;
  for (long i = 0; i < maxsteps; i++) {
    action = PyLong_FromLongLong(rand() % 6); //new
    res = pycall1(stepfn, action);  //res is new, ref to action is stolen
    done = PyTuple_GetItem(res, 2); //borrowed, no need to decref
    if (PyObject_IsTrue(done)){
      printf("done and dustid after %lu steps\n", i);
      return i;
    }
    Py_DECREF(res);
  }
  printf("mexed out (%ld steps bru)\n", maxsteps);
  return maxsteps;
}

//reset the env
void reset(PyObject* env){
  pycall0(PyObject_GetAttrString(env, "reset"));
}


//gcc pycapi.c -lpython3.6m -I /opt/local/Library/Frameworks/Python.framework/Versions/3.6/include/python3.6m -L /opt/local/Library/Frameworks/Python.framework/Versions/3.6/lib
