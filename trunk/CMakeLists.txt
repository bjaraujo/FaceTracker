
#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 2.6)

project(FaceTracker)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

add_subdirectory(src)


