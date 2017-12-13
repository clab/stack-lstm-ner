Minimum Requirements
======================
1. Cmake 2.8.7
2. [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
3. `dynet` (Don't have to install this separately; will be installed below)

Installation Steps
======================
These steps have been tested on Ubuntu 16.04 and macOS Sierra.

1. Once this repository has been cloned, the `dynet/` submodule needs to be synced

  ```bash
  git submodule init
  git submodule update
  ```
2. This will download the required files to dynet directory. Let this directory be `PATH_TO_DYNET`.
  
  ```bash
  PATH_TO_DYNET=<your_stack_lstm_dir>/dynet/
  ```
3. Download the C++ library `eigen` which is used by dynet:
  
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
6. Go back to `dynet` directory in `stack-lstm-ner` and build `dynet`. Modify the code below with your `eigen3` `include` location and boost location. 

  ```bash
  cd $PATH_TO_DYNET
  mkdir build
  cd build
  cmake .. -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3
  make -j 2
  ```
**Note:** If DYNET fails to compile and throws an error like this:
  ```bash
  $ make -j 2
  Scanning dependencies of target dynet
  Scanning dependencies of target dynet_shared
  [  1%] [  2%] Building CXX object dynet/CMakeFiles/dynet.dir/cfsm-builder.cc.o
  Building CXX object dynet/CMakeFiles/dynet_shared.dir/cfsm-builder.cc.o
  In file included from /home/user/dynet/dynet/dynet.h:13:0,
                   from /home/user/dynet/dynet/cfsm-builder.h:6,
                   from /home/user/dynet/dynet/cfsm-builder.cc:1:
  /home/user/dynet/dynet/tensor.h:22:42: fatal error: unsupported/Eigen/CXX11/Tensor: No such file or directory
  #include <unsupported/Eigen/CXX11/Tensor>
                                            ^
  compilation terminated.
  ```
Then, download and install a stable version of Eigen and rebuild DyNet:

  ```bash
  cd $HOME
  wget u.cs.biu.ac.il/~yogo/eigen.tgz
  tar zxvf eigen.tgz
  cd eigen
  ```
Repeat step 4 and run:

  ```bash
  cd $PATH_TO_DYNET/build
  rm -rf *
  ```
Now, rebuild DyNet again.
7. Create a `build` directory in <stack_lstm_dir> and in the same directory `stack-lstm-ner`, do `cmake .  -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3` and then `make`. This will build `lstm-parse` in `ner-system`

Debugging build errors
========================
If you want to see the compile commands that are used, you can run

```bash
make VERBOSE=1
```
