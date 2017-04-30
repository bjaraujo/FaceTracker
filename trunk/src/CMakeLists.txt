
project(FaceTracker)

set(FACETRACKER_HEADERS
ht/HeadTracker.h
lib/CLM.h
lib/FCheck.h
lib/FDet.h
lib/IO.h
lib/Patch.h
lib/PAW.h
lib/PDM.h
lib/Tracker.h
)

set(FACETRACKER_SOURCES
ht/HeadTracker.cpp
lib/CLM.cc
lib/FCheck.cc
lib/FDet.cc
lib/IO.cc
lib/Patch.cc
lib/PAW.cc
lib/PDM.cc
lib/Tracker.cc
exe/face_tracker.cc
)

find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLUT)

source_group("Header Files" FILES ${FACETRACKER_HEADERS})
source_group("Source Files" FILES ${FACETRACKER_SOURCES})

message(${OpenCV_DIR})

include_directories(
${OPENGL_INCLUDE_DIRS}
${GLUT_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIR}
./ht
./lib
)

add_executable(FaceTracker ${FACETRACKER_HEADERS} ${FACETRACKER_SOURCES})

target_link_libraries(FaceTracker 
${OPENGL_LIBRARIES}
${GLUT_LIBRARY}
${OpenCV_LIBRARIES}
)




