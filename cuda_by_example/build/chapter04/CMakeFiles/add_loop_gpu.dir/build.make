# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hgj/Project/mypro/mycuda/cuda_by_example

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hgj/Project/mypro/mycuda/cuda_by_example/build

# Include any dependencies generated for this target.
include chapter04/CMakeFiles/add_loop_gpu.dir/depend.make

# Include the progress variables for this target.
include chapter04/CMakeFiles/add_loop_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include chapter04/CMakeFiles/add_loop_gpu.dir/flags.make

chapter04/CMakeFiles/add_loop_gpu.dir/add_loop_gpu_generated_add_loop_gpu.cu.o: chapter04/CMakeFiles/add_loop_gpu.dir/add_loop_gpu_generated_add_loop_gpu.cu.o.depend
chapter04/CMakeFiles/add_loop_gpu.dir/add_loop_gpu_generated_add_loop_gpu.cu.o: chapter04/CMakeFiles/add_loop_gpu.dir/add_loop_gpu_generated_add_loop_gpu.cu.o.cmake
chapter04/CMakeFiles/add_loop_gpu.dir/add_loop_gpu_generated_add_loop_gpu.cu.o: ../chapter04/add_loop_gpu.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/hgj/Project/mypro/mycuda/cuda_by_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object chapter04/CMakeFiles/add_loop_gpu.dir/add_loop_gpu_generated_add_loop_gpu.cu.o"
	cd /home/hgj/Project/mypro/mycuda/cuda_by_example/build/chapter04/CMakeFiles/add_loop_gpu.dir && /usr/local/bin/cmake -E make_directory /home/hgj/Project/mypro/mycuda/cuda_by_example/build/chapter04/CMakeFiles/add_loop_gpu.dir//.
	cd /home/hgj/Project/mypro/mycuda/cuda_by_example/build/chapter04/CMakeFiles/add_loop_gpu.dir && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/hgj/Project/mypro/mycuda/cuda_by_example/build/chapter04/CMakeFiles/add_loop_gpu.dir//./add_loop_gpu_generated_add_loop_gpu.cu.o -D generated_cubin_file:STRING=/home/hgj/Project/mypro/mycuda/cuda_by_example/build/chapter04/CMakeFiles/add_loop_gpu.dir//./add_loop_gpu_generated_add_loop_gpu.cu.o.cubin.txt -P /home/hgj/Project/mypro/mycuda/cuda_by_example/build/chapter04/CMakeFiles/add_loop_gpu.dir//add_loop_gpu_generated_add_loop_gpu.cu.o.cmake

# Object files for target add_loop_gpu
add_loop_gpu_OBJECTS =

# External object files for target add_loop_gpu
add_loop_gpu_EXTERNAL_OBJECTS = \
"/home/hgj/Project/mypro/mycuda/cuda_by_example/build/chapter04/CMakeFiles/add_loop_gpu.dir/add_loop_gpu_generated_add_loop_gpu.cu.o"

bin/add_loop_gpu: chapter04/CMakeFiles/add_loop_gpu.dir/add_loop_gpu_generated_add_loop_gpu.cu.o
bin/add_loop_gpu: chapter04/CMakeFiles/add_loop_gpu.dir/build.make
bin/add_loop_gpu: /usr/lib/x86_64-linux-gnu/libcudart_static.a
bin/add_loop_gpu: /usr/lib/x86_64-linux-gnu/librt.so
bin/add_loop_gpu: chapter04/CMakeFiles/add_loop_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hgj/Project/mypro/mycuda/cuda_by_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/add_loop_gpu"
	cd /home/hgj/Project/mypro/mycuda/cuda_by_example/build/chapter04 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/add_loop_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
chapter04/CMakeFiles/add_loop_gpu.dir/build: bin/add_loop_gpu

.PHONY : chapter04/CMakeFiles/add_loop_gpu.dir/build

chapter04/CMakeFiles/add_loop_gpu.dir/clean:
	cd /home/hgj/Project/mypro/mycuda/cuda_by_example/build/chapter04 && $(CMAKE_COMMAND) -P CMakeFiles/add_loop_gpu.dir/cmake_clean.cmake
.PHONY : chapter04/CMakeFiles/add_loop_gpu.dir/clean

chapter04/CMakeFiles/add_loop_gpu.dir/depend: chapter04/CMakeFiles/add_loop_gpu.dir/add_loop_gpu_generated_add_loop_gpu.cu.o
	cd /home/hgj/Project/mypro/mycuda/cuda_by_example/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hgj/Project/mypro/mycuda/cuda_by_example /home/hgj/Project/mypro/mycuda/cuda_by_example/chapter04 /home/hgj/Project/mypro/mycuda/cuda_by_example/build /home/hgj/Project/mypro/mycuda/cuda_by_example/build/chapter04 /home/hgj/Project/mypro/mycuda/cuda_by_example/build/chapter04/CMakeFiles/add_loop_gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : chapter04/CMakeFiles/add_loop_gpu.dir/depend

