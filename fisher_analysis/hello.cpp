 // hello.cpp
 #include <iostream>
 #include <boost/python.hpp>
 
 void sayHello()
 {
   std::cout << "Hello, Python!\n";
 }
 
 BOOST_PYTHON_MODULE(hello)  // Name here must match the name of the final shared library, i.e. hello.so
 {
    boost::python::def("sayHello", &sayHello);
 }