cmake_minimum_required(VERSION 2.8)
project(IGAC)

find_package(PkgConfig)
find_package(OpenCL)
find_package(glfw3)
pkg_check_modules(GLEW glew)
pkg_check_modules(LIBTIFF libtiff-4)
pkg_check_modules(TCLAP tclap)
pkg_check_modules(X11 x11)
pkg_check_modules(Xxf86vm xxf86vm)

add_executable(IGAC
src/cg_compute.cpp
src/cg_compute.hpp
src/cg_defines.hpp
src/cg_headers.cpp
src/cg_headers.hpp
src/cg_image.cpp
src/cg_image.hpp
src/cg_project.cpp
src/cg_renderer.cpp
src/cg_renderer.hpp
src/cg_active_contour.cpp
src/cg_active_contour.hpp
)

include_directories(include)
set_property(TARGET IGAC PROPERTY CXX_STANDARD 14)

target_link_libraries(IGAC glfw)
target_link_libraries(IGAC ${GLEW_LIBRARIES})
target_link_libraries(IGAC ${LIBTIFF_LIBRARIES})
target_link_libraries(IGAC ${TCLAP_LIBRARIES})
target_link_libraries(IGAC ${OpenCL_LIBRARIES})
target_link_libraries(IGAC ${X11_LIBRARIES})
target_link_libraries(IGAC ${Xxf86vm_LIBRARIES})
target_link_libraries(IGAC libpthread.so)
target_link_libraries(IGAC libdl.so)
