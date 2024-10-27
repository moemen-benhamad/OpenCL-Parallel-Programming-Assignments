cmake_minimum_required(VERSION 3.5)
project(base)
set (CMAKE_CXX_STANDARD 11)

include_directories(
  lib/
  common/
)

link_directories(
	${PROJECT_SOURCE_DIR}/OpenCL-ICD-Loader/inc
	${PROJECT_SOURCE_DIR}/OpenCL-ICD-Loader/inc/CL
)

IF (WIN32)
  SET(CL_LIBRARIES OpenCL)
ELSE (WIN32)
  SET(CL_LIBRARIES OpenCL)
ENDIF(WIN32)

add_definitions(-DSRC_PATH=\"${CMAKE_SOURCE_DIR}/\")
add_executable(closest_pair closest_pair.cpp closest_pair.cl common/clutils.cpp)
if(APPLE)
	target_link_libraries(closest_pair "-framework OpenCL" )
else()
	target_link_libraries(closest_pair ${CL_LIBRARIES})
endif()

add_subdirectory(OpenCL-ICD-Loader)
