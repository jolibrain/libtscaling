find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/tscaling.i")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_custom_command(OUTPUT ${OUTPUT}
  COMMAND ${SWIG_EXECUTABLE} -I${CMAKE_SOURCE_DIR}/include -python -c++ -o _tscaling_module.cc ${CMAKE_CURRENT_SOURCE_DIR}/tscaling.i
  COMMAND ${PYTHON} ${SETUP_PY} build_ext --inplace
  COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
  DEPENDS ${DEPS})

add_custom_target(target ALL DEPENDS ${OUTPUT})

install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
