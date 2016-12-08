Minimum Requirements
======================
1. Boost 1.58.0
2. Cmake 2.8.7
3. [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
4. pycnn (don't have to install separately; will be installed below)

Installation Steps
======================
These steps have been tested on Ubuntu 16.04 and macOS Sierra.

1. Once this repository has been cloned, the `cnn/` submodule needs to be synced

  ```bash
  git submodule init
  git submodule update
  ```
2. This will download the required files to cnn directory. Let this directory be `PATH_TO_CNN`.
  
  ```bash
  PATH_TO_CNN=<your_stack_lstm_dir>/cnn/
  ```
3. Download the C++ library `eigen` which is used by cnn:
  
  ```bash
  cd $HOME
  hg clone https://bitbucket.org/eigen/eigen/
  cd eigen
  ```
  **Note:** There were compilation issues with some versions of `eigen`. This installation has been successful with `Eigen v3.3.1`
4. Now, create a `build` directory and install eigen:

  ```bash
  mkdir build
  cd build
  cmake ..
  ```
5. Run `sudo make install`. This will push the library files to the local `include` directory. On Ubuntu 16.04 and macOS Sierra, they are copied to `/usr/local/include/eigen3`. 
6. Go back to `cnn` directory in `stack-lstm-ner` and build `cnn`. Modify the code below with your `eigen3` `include` location and boost location. 

  ```bash
  cd $PATH_TO_CNN
  mkdir build
  cd build
  cmake .. -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3 -DBOOST_ROOT=$HOME/.local/boost_1_58_0 -DBoost_NO_BOOST_CMAKE=ON
  make -j 2
  ```
Your program might work without providing the boost location as a command line argument. **Note:** If CNN fails to compile and throws an error like this:
  ```bash
  $ make -j 2
  Scanning dependencies of target cnn
  Scanning dependencies of target cnn_shared
  [  1%] [  2%] Building CXX object cnn/CMakeFiles/cnn.dir/cfsm-builder.cc.o
  Building CXX object cnn/CMakeFiles/cnn_shared.dir/cfsm-builder.cc.o
  In file included from /home/user/cnn/cnn/cnn.h:13:0,
                   from /home/user/cnn/cnn/cfsm-builder.h:6,
                   from /home/user/cnn/cnn/cfsm-builder.cc:1:
  /home/user/cnn/cnn/tensor.h:22:42: fatal error: unsupported/Eigen/CXX11/Tensor: No such file or directory
  #include <unsupported/Eigen/CXX11/Tensor>
                                            ^
  compilation terminated.
  ```
Then, download and install a stable version of Eigen and rebuild CNN:

  ```bash
  cd $HOME
  wget u.cs.biu.ac.il/~yogo/eigen.tgz
  tar zxvf eigen.tgz
  cd eigen
  ```
Repeat step 4 and run:

  ```bash
  cd $PATH_TO_CNN/build
  rm -rf *
  ```
Now, rebuild CNN again.
7. Install `cython` using `pip install cython` as this is required for compiling `pycnn`.
8. Go to `pycnn` in `cnn` directory and customize `setup.py` to include (i) parent directory where main `cnn` directory is saved and (ii) the path of your local `include` `eigen3`. Then run:

  ```bash
  make
  make install
  ```
If `make install` throws an error like:
  ```bash
  cp ../build/cnn/libcnn_shared.dylib .
  python setup.py build_ext --inplace
  running build_ext
  skipping 'pycnn.cpp' Cython extension (up-to-date)
  python setup.py install --user
  running install
  error: can't combine user with prefix, exec_prefix/home, or install_(plat)base
  make: *** [install] Error 1
  ```
Then, in `install` target in `makefile`, add ` --prefix=` (_yes_, it is blank) at the end of the line `${PYTHON} setup.py install --user`. Now, run `make install`
8. To verify the installation, run the command below in `$PATH_TO_CNN/build`

  ```bash
  ./examples/xor
  ```
This will train a multi layer perceptron to predict the xor function.
9. Create a `build` directory in <stack_lstm_dir> and in the same directory `stack-lstm-ner`, do `cmake .` and then `make`. This will build `lstm-parse` in `ner-system`

Debugging build errors
========================
If you want to see the compile commands that are used, you can run

```bash
make VERBOSE=1
```
