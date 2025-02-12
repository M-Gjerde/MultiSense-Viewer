# CMAKE FILE DESCRIPTION
# - Sets build type to Release if none was specified
# - Pulls submodules into /external folder
# - Adds each submodule to project add_subdirectory/include_directory
# - Adds internal library Autoconnect
# - Option to enabled all warnings when compiling with GCC

set(CRL_SERVER_IP 35.211.65.110:80)
set(CRL_SERVER_PROTOCOL http)
set(CRL_SERVER_DESTINATION /api.php)
set(CRL_SERVER_VERSIONINFO_DESTINATION /version.php)
set(VIEWER_LOG_LEVEL LOG_INFO)

find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" OFF)
    if (GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if (NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif ()
    endif ()
endif ()

# Include Submodules into project.
# Check if exists or display fatal error
set(GLM_DIR external/glm)
set(GLFW_DIR external/glfw)
set(TINYGLFT_DIR external/tinygltf)
set(FMT_DIR external/fmt)
set(LIBTIFF_DIR external/libtiff)
set(LIBMULTISENSE_DIR external/LibMultiSense)
set(SIMPLEINI_DIR external/simpleini)
set(IMGUI_DIR external/imgui)
set(KTX_DIR external/KTX-Software)
set(NLOHMANN_JSON external/json)
set(CPP_HTTPLIB external/cpp-httplib)

set(ROSBAGS_WRITER_DIR internal/rosbag_writer_cpp)
set(AUTOCONNECT_DIR internal/AutoConnect)


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${GLM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${GLM_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding GLM from directory: ${GLM_DIR}")

    include_directories(SYSTEM ${GLM_DIR})

endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${GLFW_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${GLFW_DIR}/ not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding GLFW from directory: ${GLFW_DIR}")

    add_subdirectory(${GLFW_DIR} glfw_binary EXCLUDE_FROM_ALL)
    include_directories(SYSTEM ${GLFW_DIR}/include)
    include_directories(SYSTEM ${GLFW_DIR}/deps)
endif ()


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${TINYGLFT_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${TINYGLFT_DIR}/ not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding TINYGLTF from directory: ${TINYGLFT_DIR}")

    set(TINYGLTF_HEADER_ONLY ON CACHE INTERNAL "" FORCE)
    set(TINYGLTF_INSTALL OFF CACHE INTERNAL "" FORCE)
    include_directories(SYSTEM ${TINYGLFT_DIR})
endif ()


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${LIBTIFF_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${TINYTIFF_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding libtiff from directory: ${LIBTIFF_DIR}")
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(tiff-tools OFF)
    set(tiff-tests OFF)
    set(tiff-contrib OFF)
    set(tiff-docs OFF)
    add_subdirectory(${LIBTIFF_DIR})
    include_directories(${LIBTIFF_DIR}/libtiff)
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${LIBMULTISENSE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${LIBMULTISENSE_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding LibMultiSense from directory: ${LIBMULTISENSE_DIR}")
    set(MULTISENSE_BUILD_UTILITIES OFF)
    if (WIN32)
        set(BUILD_SHARED_LIBS ON)
    endif ()
    include_directories(SYSTEM ${LIBMULTISENSE_DIR}/source/LibMultiSense)
    add_subdirectory(${LIBMULTISENSE_DIR}/source/LibMultiSense)
    if (WIN32)
        set(BUILD_SHARED_LIBS OFF)
    endif ()

endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${SIMPLEINI_DIR}/SimpleIni.h")
    message(FATAL_ERROR "The submodules ${SIMPLEINI_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding SIMPLEINI from directory: ${SIMPLEINI_DIR}")

    set(SimpleIni_SRC ${SIMPLEINI_DIR}/SimpleIni.h ${SIMPLEINI_DIR}/ConvertUTF.c ${SIMPLEINI_DIR}/ConvertUTF.h)
    add_library(SimpleIni STATIC ${SimpleIni_SRC})
    include_directories(SYSTEM ${SIMPLEINI_DIR})
    set_target_properties(SimpleIni PROPERTIES LINKER_LANGUAGE CXX)
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${IMGUI_DIR}/imgui.h")
    message(FATAL_ERROR "The submodules ${IMGUI_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding IMGUI from directory: ${IMGUI_DIR}")

    set(IMGUI_DIR external/imgui)
    set(IMGUI_SOURCES
            ${IMGUI_DIR}/imgui.h
            ${IMGUI_DIR}/imgui_internal.h
            ${IMGUI_DIR}/imgui.cpp
            ${IMGUI_DIR}/imgui_demo.cpp
            ${IMGUI_DIR}/imgui_draw.cpp
            ${IMGUI_DIR}/imgui_tables.cpp
            ${IMGUI_DIR}/imgui_widgets.cpp
    )
    set(IMGUI_INCLUDE ${IMGUI_DIR}/
            ${IMGUI_DIR}/backends
            ${PROJECT_SOURCE_DIR}/include/Viewer/ImGui/Custom
    )
    target_sources(${PROJECT_NAME} PRIVATE
            ${IMGUI_SOURCES}
    )
    target_include_directories(${PROJECT_NAME} PRIVATE
            ${IMGUI_INCLUDE})

endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${AUTOCONNECT_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${AUTOCONNECT_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding AUTOCONNECT from directory: ${AUTOCONNECT_DIR}")
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(BUILD_STANDALONE OFF)
    add_subdirectory(${AUTOCONNECT_DIR})
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${ROSBAGS_WRITER_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${ROSBAGS_WRITER_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding ROSBAGS_WRITER_DIR from directory: ${ROSBAGS_WRITER_DIR}")
    set(BUILD_TESTS OFF CACHE BOOL "Build tests for rosbag_writer_cpp" FORCE)
    add_subdirectory(${ROSBAGS_WRITER_DIR})
    target_include_directories(rosbag_cpp_writer PUBLIC ${ROSBAGS_WRITER_DIR}/include)

    message("[INFO] Including rosbags_writer directory: ${ROSBAGS_WRITER_DIR}/include")

endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${FMT_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${FMT_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding FMT from directory: ${FMT_DIR}")
    include_directories(SYSTEM ${FMT_DIR}/include)
    add_subdirectory(${FMT_DIR})
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${KTX_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${KTX_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding KTX from directory: ${KTX_DIR}")
    set(KTX_FEATURE_STATIC_LIBRARY ON)
    set(KTX_FEATURE_TESTS OFF)
    set(KTX_FEATURE_TOOLS OFF)
    add_subdirectory(${KTX_DIR})
    include_directories(SYSTEM ${KTX_DIR}/include)
    link_directories(${KTX_DIR}/lib)

endif ()


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${NLOHMANN_JSON}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${NLOHMANN_JSON} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding NLOHMANN_JSON from directory: ${NLOHMANN_JSON}")
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install OFF CACHE INTERNAL "")
    add_subdirectory(${NLOHMANN_JSON})
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${CPP_HTTPLIB}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${CPP_HTTPLIB} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding CPP_HTTPLIB from directory: ${CPP_HTTPLIB}")
    set(HTTPLIB_REQUIRE_OPENSSL OFF)
    set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF)
    set(OPENSSL_USE_STATIC_LIBS ON)
    add_subdirectory(${CPP_HTTPLIB})
endif ()

# ExportScriptIncludes Generates ScriptHeader.h and Scripts.txt for automatic import of the script functionality in the viewer.
function(ExportScriptIncludes)
    string(TIMESTAMP Today)
    file(GLOB_RECURSE SCRIPT_HEADERS RELATIVE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/Viewer/Scripts/*.h")
    list(FILTER SCRIPT_HEADERS EXCLUDE REGEX "/Private/")
    file(WRITE ${CMAKE_SOURCE_DIR}/Assets/Generated/ScriptHeader.h "// Generated from Cmake ${Today} \n")
    file(WRITE ${CMAKE_SOURCE_DIR}/Assets/Generated/Scripts.txt "# Generated from Cmake ${Today} \n")
    foreach (Src ${SCRIPT_HEADERS})
        file(APPEND ${CMAKE_SOURCE_DIR}/Assets/Generated/ScriptHeader.h "\#include \"${Src}\"\n")
    endforeach (Src ${SCRIPT_HEADERS})

    foreach (Src ${SCRIPT_HEADERS})
        string(REGEX MATCH "[^\\/]+$" var ${Src})
        string(REGEX MATCH "^[^.]+" res ${var})
        file(APPEND ${CMAKE_SOURCE_DIR}/Assets/Generated/Scripts.txt ${res} \n)
    endforeach (Src ${SCRIPT_HEADERS})
endfunction()

function(GenerateVersionFile)
    file(WRITE ${CMAKE_SOURCE_DIR}/Assets/Generated/VersionInfo "VERSION=${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}\nSERVER=${CRL_SERVER_IP}\nPROTOCOL=${CRL_SERVER_PROTOCOL}\nDESTINATION=${CRL_SERVER_DESTINATION}\nDESTINATION_VERSIONINFO=${CRL_SERVER_VERSIONINFO_DESTINATION}\nLOG_LEVEL=${VIEWER_LOG_LEVEL}")
endfunction()

if (UNIX)
    set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/opt/multisense)
elseif (WIN32)
    set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/)
endif ()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "Set install directory to ${INSTALL_DIRECTORY}")
    set(${CMAKE_INSTALL_PREFIX} ${INSTALL_DIRECTORY}
            CACHE PATH "default install path" FORCE)
endif ()
