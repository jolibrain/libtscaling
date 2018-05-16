cmake_minimum_required(VERSION 3.0)
project (tscaling)
include_directories(include)
file(GLOB SOURCES "src/tscaling.cpp")
file(GLOB TEST_SOURCES "src/test.cpp")

OPTION(FORCE_LBFGSPP "force use bundled LBFGSPP instead of system LBFGS")
OPTION(BUILD_TEST "build test app")

find_library(LBFGS_LIB lbfgs)
if (NOT LBFGS_LIB OR FORCE_LBFGSPP)
   set(USE_LBFGSPP ON)
else()
   set(USE_LBFGSPP OFF) 
endif()



if (BUILD_TEST)
  add_executable(test_tscaling ${TEST_SOURCES})
endif()

add_library(tscaling SHARED ${SOURCES})

set (CMAKE_CXX_FLAGS "-g -Wall -std=c++11")

if (USE_LBFGSPP)
  find_package (Eigen3)
  add_definitions(-DLBFGSPP)
  include_directories(${EIGEN3_INCLUDE_DIR})
else()
   target_link_libraries(tscaling ${LBFGS_LIB}) 
endif()

if (BUILD_TEST)
   target_link_libraries(test_tscaling tscaling)       
endif()


