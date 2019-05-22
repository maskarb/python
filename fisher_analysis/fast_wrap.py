#include<boost/python.hpp>

BOOST_PYTHON_MODULE(FastFastUnivariateDensityDerivative)  // Name here must match the name of the final shared library, i.e. mantid.dll or mantid.so
 {
    class_<FastUnivariateDensityDerivative>("FastUnivariateDensityDerivative")
        .def(init<int, int, double, double, double, int, double, double>())
        .def("evaluate", &FastUnivariateDensityDerivative::Evaluate)
 }