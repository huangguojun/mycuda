# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/hgj/Support/cmake-3.8.2-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/hgj/Support/cmake-3.8.2-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hgj/Project/cuda/cuda_by_example

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hgj/Project/cuda/cuda_by_example/build

# Include any dependencies generated for this target.
include chapter03/CMakeFiles/simple_kernel_params.dir/depend.make

# Include the progress variables for this target.
include chapter03/CMakeFiles/simple_kernel_params.dir/progress.make

# Include the compile flags for this target's objects.
include chapter03/CMakeFiles/simple_kernel_params.dir/flags.make

chapter03/CMakeFiles/simple_kernel_params.dir/simple_kernel_params_generated_simple_kernel_params.cu.o: chapter03/CMakeFiles/simple_kernel_params.dir/simple_kernel_params_generated_simple_kernel_params.cu.o.depend
chapter03/CMakeFiles/simple_kernel_params.dir/simple_kernel_params_generated_simple_kernel_params.cu.o: chapter03/CMakeFiles/simple_kernel_params.dir/simple_kernel_params_generated_simple_kernel_params.cu.o.cmake
chapter03/CMakeFiles/simple_kernel_params.dir/simple_kernel_params_generated_simple_kernel_params.cu.o: ../chapter03/simple_kernel_params.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/hgj/Project/cuda/cuda_by_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object chapter03/CMakeFiles/simple_kernel_params.dir/simple_kernel_params_generated_simple_kernel_params.cu.o"
	cd /home/hgj/Project/cuda/cuda_by_example/build/chapter03/CMakeFiles/simple_kernel_params.dir && /home/hgj/Support/cmake-3.8.2-Linux-x86_64/bin/cmake -E make_directory /home/hgj/Project/cuda/cuda_by_example/build/chapter03/CMakeFiles/simple_kernel_params.dir//.
	cd /home/hgj/Project/cuda/cuda_by_example/build/chapter03/CMakeFiles/simple_kernel_params.dir && /home/hgj/Support/cmake-3.8.2-Linux-x86_64/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/hgj/Project/cuda/cuda_by_example/build/chapter03/CMakeFiles/simple_kernel_params.dir//./simple_kernel_params_generated_simple_kernel_params.cu.o -D generated_cubin_file:STRING=/home/hgj/Project/cuda/cuda_by_example/build/chapter03/CMakeFiles/simple_kernel_params.dir//./simple_kernel_params_generated_simple_kernel_params.cu.o.cubin.txt -P /home/hgj/Project/cuda/cuda_by_example/build/chapter03/CMakeFiles/simple_kernel_params.dir//simple_kernel_params_generated_simple_kernel_params.cu.o.cmake

# Object files for target simple_kernel_params
simple_kernel_params_OBJECTS =

# External object files for target simple_kernel_params
simple_kernel_params_EXTERNAL_OBJECTS = \
"/home/hgj/Project/cuda/cuda_by_example/build/chapter03/CMakeFiles/simple_kernel_params.dir/simple_kernel_params_generated_simple_kernel_params.cu.o"

bin/simple_kernel_params: chapter03/CMakeFiles/simple_kernel_params.dir/simple_kernel_params_generated_simple_kernel_params.cu.o
bin/simple_kernel_params: chapter03/CMakeFiles/simple_kernel_params.dir/build.make
bin/simple_kernel_params: /usr/lib/x86_64-linux-gnu/libcudart_static.a
bin/simple_kernel_params: /usr/lib/x86_64-linux-gnu/librt.so
bin/simple_kernel_params: chapter03/CMakeFiles/simple_kernel_params.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hgj/Project/cuda/cuda_by_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/simple_kernel_params"
	cd /home/hgj/Project/cuda/cuda_by_example/build/chapter03 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simple_kernel_params.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
chapter03/CMakeFiles/simple_kernel_params.dir/build: bin/simple_kernel_params

.PHONY : chapter03/CMakeFiles/simple_kernel_params.dir/build

chapter03/CMakeFiles/simple_kernel_params.dir/requires:

.PHONY : chapter03/CMakeFiles/simple_kernel_params.dir/requires

chapter03/CMakeFiles/simple_kernel_params.dir/clean:
	cd /home/hgj/Project/cuda/cuda_by_example/build/chapter03 && $(CMAKE_COMMAND) -P CMakeFiles/simple_kernel_params.dir/cmake_clean.cmake
.PHONY : chapter03/CMakeFiles/simple_kernel_params.dir/clean

chapter03/CMakeFiles/simple_kernel_params.dir/depend: chapter03/CMakeFiles/simple_kernel_params.dir/simple_kernel_params_generated_simple_kernel_params.cu.o
	cd /home/hgj/Project/cuda/cuda_by_example/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hgj/Project/cuda/cuda_by_example /home/hgj/Project/cuda/cuda_by_example/chapter03 /home/hgj/Project/cuda/cuda_by_example/build /home/hgj/Project/cuda/cuda_by_example/build/chapter03 /home/hgj/Project/cuda/cuda_by_example/build/chapter03/CMakeFiles/simple_kernel_params.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : chapter03/CMakeFiles/simple_kernel_params.dir/depend

