cmake_minimum_required(VERSION 3.0)
project (tscaling)
include_directories(include)
file(GLOB SOURCES "src/*.cpp")

OPTION(USE_LBFGSPP "use bundled LBFGSPP instead of system LBFGS")
OPTION(BUILD_TEST "build test app")

find_library(LBFGS_LIB lbfgs)

if (BUILD_TEST)
  add_executable(test_tscaling ${SOURCES})
endif()

add_library(tscaling SHARED ${SOURCES})

set (CMAKE_CXX_FLAGS "-Wall -std=c++11")

if (USE_LBFGSPP)
  find_package (Eigen3)
  add_definitions(-DLBFGSPP)
  include_directories(${EIGEN3_INCLUDE_DIR})
else()
  if (BUILD_TEST)
    target_link_libraries(test_tscaling ${LBFGS_LIB})
  endif()
  target_link_libraries(tscaling ${LBFGS_LIB})
endif()


