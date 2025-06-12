**Basic structure of the project**

The core of this project is a C++ header called Ising.hpp. This defines a object that can do basic monte carlo simulation of afm Ising model and record thermaldynamical data. 

The class is exposed to python using pybind11. The file bind.cpp records how it is exposed to python. 

The file wrapper.py processes the recorded data, and output directly the mean value and estimated error, based on raw data. 


**How to create a C++ script avalilable for python?**

Step 0: Download CMake.

Step 1: copy pybind11 folder under extern folder. You can find the latest release in github

Step 2: Create an empty folder "build"

Step 3: Copy CMakeLists.txt

Step 4: run the following code in cmd: 

*For Windows,*

> cd build
> 
> cmake ..
> 
> cmake --build . --config Release

*For Mac/Linux,*

> cd build
> 
> cmake .. -DCMAKE_BUILD_TYPE=Release
> 
> make

Now you can find the script in the build folder
