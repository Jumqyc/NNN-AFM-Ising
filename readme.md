

**How to create a python module using C++?**

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
