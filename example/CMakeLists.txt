cmake_minimum_required(VERSION 3.10)

project("main")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-D__STDC_CONSTANT_MACROS")
add_compile_options("/EHsc")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if (SUPRESS_JSON)
    message("-- IDE configuration setttings are disabled")
else()
    message("-- IDE configuration settings will be written to .vscode/c_cpp_properties.json")
    set(json_filename ${CMAKE_CURRENT_SOURCE_DIR}/.vscode/c_cpp_properties.json)
    set(json_string "{\n\t\"configurations\": [\n\t\t{\n\t\t\t\"name\": \"Win32\",\n\t\t\t\"includePath\": [\n\t\t\t\t\"\${workspaceFolder}/**\",")
endif()

if (OpenCV_INCLUDE_DIRS AND OpenCV_LIBS)
    cmake_path(NORMAL_PATH OpenCV_INCLUDE_DIRS OUTPUT_VARIABLE OpenCV_INCLUDE_DIRS)
    cmake_path(NORMAL_PATH OpenCV_LIBS OUTPUT_VARIABLE OpenCV_LIBS)
    message("-- OpenCV is using include dirs ${OpenCV_INCLUDE_DIRS} and libs ${OpenCV_LIBS}")
else()
    find_package(OpenCV)
    if (OpenCV_FOUND)
        message("-- Found OpenCV")
    else()
        message("-- **WARNING** Unable to find OpenCV automatically, this may be due to the lack of OpenCV_DIR environment variable. You could also try setting -DOpenCV_INCLUDE_DIRS and -DOpenCV_LIBS locations manually")
    endif()
endif()
if (NOT SUPRESS_JSON AND OpenCV_INCLUDE_DIRS)
    set(json_opencv_include ${OpenCV_INCLUDE_DIRS})
    list(TRANSFORM json_opencv_include PREPEND "\n\t\t\t\t\"") 
    list(TRANSFORM json_opencv_include APPEND "\",")
    string(APPEND json_string ${json_opencv_include})
    #message("json_include: ${json_opencv_include}")
endif()

if (Eigen3_INCLUDE_DIRS)
    message("-- Eigen3 is using include dirs ${Eigen3_INCLUDE_DIRS}")
else()
    find_package(Eigen3)
    if (Eigen3_FOUND)
        set(Eigen3_LIBS Eigen3::Eigen)
        message("-- Found Eigen3: ${Eigen3_DIR}")
    else()
        message("-- **WARNING** Unable to find Eigen3 automatically, use -DEigen3_INCLUDE_DIRS to set location manually")
    endif()
endif()
if (NOT SUPRESS_JSON)
get_target_property(json_eigen3_include Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
list(TRANSFORM json_eigen3_include PREPEND "\n\t\t\t\t\"") 
list(TRANSFORM json_eigen3_include APPEND "\",")
string(APPEND json_string ${json_eigen3_include})
#message("json_include: ${json_eigen3_include}")
endif()

set(Tensorflow_LIBRARIES C:/Users/sr996/source/tensorflow/2.7.0/lib/tensorflow.lib)
set(Tensorflow_INCLUDE_DIRS C:/Users/sr996/source/tensorflow/2.7.0/include)
if (NOT SUPRESS_JSON)
    set(json_tensorflow_include ${Tensorflow_INCLUDE_DIRS})
    list(TRANSFORM json_tensorflow_include PREPEND "\n\t\t\t\t\"") 
    list(TRANSFORM json_tensorflow_include APPEND "\"")
    string(APPEND json_string ${json_tensorflow_include})
    #message("json_include: ${json_darknet_include}")
endif()

#set(DeepSort_LIBRARIES C:/Users/sr996/source/test/libDeepSort/build/Release/libDeepSort.lib)
#set(DeepSort_INCLUDE_DIRS C:/Users/sr996/source/test/libDeepSort/include)
set(DeepSort_LIBRARIES ../../build/Release/libDeepSort)
set(DeepSort_INCLUDE_DIRS ../include)

if (NOT SUPRESS_JSON)
    string(APPEND json_string "\n\t\t\t],\n\t\t\t\"defines\": [\n\t\t\t\t\"_DEBUG\",\n\t\t\t\t\"UNICODE\",\n\t\t\t\t\"_UNICODE\"\n\t\t\t],")
    string(APPEND json_string "\n\t\t\t\"windowsSdkVersion\": \"${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}\",")
    string(APPEND json_string "\n\t\t\t\"compilerPath\": \"${CMAKE_CXX_COMPILER}\",")
    if ("${CMAKE_CXX_STANDARD}" STREQUAL "20")
        string(APPEND json_string "\n\t\t\t\"cStandard\": \"c20\",\n\t\t\t\"cppStandard\": \"c++20\",")
    endif()
    if ("${CMAKE_CXX_STANDARD}" STREQUAL "17")
        string(APPEND json_string "\n\t\t\t\"cStandard\": \"c17\",\n\t\t\t\"cppStandard\": \"c++17\",")
    endif()
    if ("${CMAKE_CXX_STANDARD}" STREQUAL "14")
        string(APPEND json_string "\n\t\t\t\"cStandard\": \"c14\",\n\t\t\t\"cppStandard\": \"c++14\",")
    endif()
    if ("${CMAKE_CXX_STANDARD}" STREQUAL "11")
        string(APPEND json_string "\n\t\t\t\"cStandard\": \"c11\",\n\t\t\t\"cppStandard\": \"c++11\",")
    endif()
    string(APPEND json_string "\n\t\t\t\"intelliSenseMode\": \"windows-msvc-x64\"\n\t\t}\n\t],\n\t\"version\": 4\n}")
    message("-- IDE configuration settings written to .vscode/c_cpp_properties.json")
endif()

file(WRITE ${json_filename} ${json_string})

if (NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    message(STATUS "This project has a top-level one called [${CMAKE_PROJECT_NAME}]")
else()
    message(STATUS "This project is a top-level one")
endif()

add_executable(main
                    main.cpp
              )

target_link_libraries(${PROJECT_NAME} PRIVATE 
                        ${OpenCV_LIBS} 
                        ${Eigen3_LIBS}
                        ${Tensorflow_LIBRARIES}
                        ${DeepSort_LIBRARIES}
                     )

target_include_directories(${PROJECT_NAME} PUBLIC
                            ${OpenCV_INCLUDE_DIRS}
                            ${Tensorflow_INCLUDE_DIRS}
                            ${DeepSort_INCLUDE_DIRS}
                          )

