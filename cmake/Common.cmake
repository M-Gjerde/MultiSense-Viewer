# CMAKE FILE DESCRIPTION
# - Sets build type to Release if none was specified
# - Pulls submodules into /external folder
# - Adds each submodule to project add_subdirectory/include_directory
# - Adds internal library Autoconnect
# - Option to enable all warnings when compiling with GCC

set(CRL_SERVER_IP "35.211.65.110:80")
set(CRL_SERVER_PROTOCOL "http")
set(CRL_SERVER_DESTINATION "/api.php")
set(CRL_SERVER_VERSIONINFO_DESTINATION "/version.php")
set(VIEWER_LOG_LEVEL "LOG_INFO")

# Find Git for submodule handling
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" OFF)
    if (GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT
        )
        if (NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif ()
    endif ()
endif ()
# Include the proprietary GigE-Vision Module
if (PROPRIETARY_GIGEVISION_MODULE)
    if (NOT GIGEVISION_HEADERS_PATH)
        message(FATAL_ERROR "GIGEVISION_HEADERS_PATH is required when PROPRIETARY_GIGEVISION_MODULE is ON")
    endif ()
    if (NOT GIGEVISION_LIB_PATH)
        message(FATAL_ERROR "GIGEVISION_LIB_PATH is required when PROPRIETARY_GIGEVISION_MODULE is ON")
    endif ()

    # Include directories for the proprietary module
    include_directories(${GIGEVISION_HEADERS_PATH})

    # Add the proprietary module library
    # Assuming the proprietary module provides a library file, you can link it like this:
    set(GIGEVISION_LIB ${GIGEVISION_LIB_PATH}/libcrlgev.a)

    # Check if the library exists
    if (NOT EXISTS ${GIGEVISION_LIB})
        message(FATAL_ERROR "Proprietary module library not found at ${GIGEVISION_LIB}")
    endif ()
    message(STATUS "[VkRenderINFO]: Proprietary GIGEVISION ENABLED")
else ()
    message(STATUS "[VkRenderINFO]: Proprietary GIGEVISION NOT ENABLED")
endif ()
# Include Submodules into project.
# Check if exists or display fatal error
set(GLM_DIR external/glm)
set(VULKAN_MEMORY_ALLOCATOR_DIR external/VulkanMemoryAllocator)
set(GLFW_DIR external/glfw)
set(TINYGLFT_DIR external/tinygltf)
set(LIBMULTISENSE_DIR external/LibMultiSense)
set(FMT_DIR external/fmt)
set(LIBTIFF_DIR external/libtiff)
set(IMGUI_DIR external/imgui)
set(KTX_DIR external/KTX-Software)
set(NLOHMANN_JSON external/json)
set(CPP_HTTPLIB external/cpp-httplib)
set(ENTT_DIR external/entt)
set(TINY_OBJ_LOADER_DIR external/tinyobjloader)
set(TINY_PLY_DIR external/tinyply)
set(YAML_CPP external/yaml-cpp)

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${GLM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${GLM_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding GLM from directory: ${GLM_DIR}")

    include_directories(SYSTEM ${GLM_DIR})

endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${VULKAN_MEMORY_ALLOCATOR_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${VULKAN_MEMORY_ALLOCATOR_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding VULKAN_MEMORY_ALLOCATOR_DIR from directory: ${VULKAN_MEMORY_ALLOCATOR_DIR}")

    add_subdirectory(${VULKAN_MEMORY_ALLOCATOR_DIR})

endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${GLFW_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${GLFW_DIR}/ not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding GLFW from directory: ${GLFW_DIR}")

    add_subdirectory(${GLFW_DIR} glfw_binary EXCLUDE_FROM_ALL)
    include_directories(SYSTEM ${GLFW_DIR}/include)
    include_directories(SYSTEM ${GLFW_DIR}/deps)
endif ()


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${TINYGLFT_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${TINYGLFT_DIR}/ not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding TINYGLTF from directory: ${TINYGLFT_DIR}")

    set(TINYGLTF_HEADER_ONLY ON CACHE INTERNAL "" FORCE)
    set(TINYGLTF_INSTALL OFF CACHE INTERNAL "" FORCE)
    include_directories(SYSTEM ${TINYGLFT_DIR})
endif ()


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${TINY_OBJ_LOADER_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${TINY_OBJ_LOADER_DIR}/ not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding TINY_OBJ_LOADER_DIR from directory: ${TINY_OBJ_LOADER_DIR}")

    add_subdirectory(${TINY_OBJ_LOADER_DIR})
    include_directories(${TINY_OBJ_LOADER_DIR})

endif ()
if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${TINY_PLY_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${TINY_PLY_DIR}/ not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding TINY_PLY_DIR from directory: ${TINY_PLY_DIR}")
    add_subdirectory(${TINY_PLY_DIR})
    include_directories("${TINY_PLY_DIR}/source")
endif ()


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${LIBTIFF_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${TINYTIFF_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding libtiff from directory: ${LIBTIFF_DIR}")
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(tiff-tools OFF)
    set(tiff-tests OFF)
    set(tiff-contrib OFF)
    set(tiff-docs OFF)
    add_subdirectory(${LIBTIFF_DIR})
    include_directories(${LIBTIFF_DIR}/libtiff)
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${IMGUI_DIR}/imgui.h")
    message(FATAL_ERROR "The submodules ${IMGUI_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding IMGUI from directory: ${IMGUI_DIR}")

    set(IMGUI_DIR external/imgui)
    set(IMPLOT3D_DIR external/implot3d)

    target_sources(${PROJECT_NAME} PRIVATE
            ${IMGUI_DIR}/imgui.h
            ${IMGUI_DIR}/imgui_internal.h
            ${IMGUI_DIR}/imgui.cpp
            ${IMGUI_DIR}/imgui_demo.cpp
            ${IMGUI_DIR}/imgui_draw.cpp
            ${IMGUI_DIR}/imgui_tables.cpp
            ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
            ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
            ${IMGUI_DIR}/imgui_widgets.cpp

            ${IMPLOT3D_DIR}/implot3d.cpp
            ${IMPLOT3D_DIR}/implot3d_demo.cpp
            ${IMPLOT3D_DIR}/implot3d_items.cpp
            ${IMPLOT3D_DIR}/implot3d_meshes.cpp

    )
    target_include_directories(${PROJECT_NAME} PRIVATE
            ${IMGUI_DIR}/
            ${IMGUI_DIR}/backends
            ../src/Viewer/Rendering/ImGui/Custom

            ${IMPLOT3D_DIR}/
    )


endif ()


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${FMT_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${FMT_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding FMT from directory: ${FMT_DIR}")
    include_directories(SYSTEM ${FMT_DIR}/include)
    add_subdirectory(${FMT_DIR})
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${LIBMULTISENSE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${LIBMULTISENSE_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding LIBMULTISENSE_DIR from directory: ${LIBMULTISENSE_DIR}")
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

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${NLOHMANN_JSON}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${NLOHMANN_JSON} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding NLOHMANN_JSON from directory: ${NLOHMANN_JSON}")
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install OFF CACHE INTERNAL "")
    add_subdirectory(${NLOHMANN_JSON})
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${CPP_HTTPLIB}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${CPP_HTTPLIB} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding CPP_HTTPLIB from directory: ${CPP_HTTPLIB}")
    set(HTTPLIB_REQUIRE_OPENSSL OFF)
    set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF)
    set(OPENSSL_USE_STATIC_LIBS ON)
    add_subdirectory(${CPP_HTTPLIB})
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${YAML_CPP}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${YAML_CPP} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding YAML_CPP from directory: ${YAML_CPP}")
    include_directories(${YAML_CPP}/include)
    add_subdirectory(${YAML_CPP})
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${ENTT_DIR}")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${ENTT_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message(STATUS "[VkRenderINFO]: Adding ENTT_DIR from directory: ${ENTT_DIR}")
    include_directories(${ENTT_DIR}/include/)
endif ()

# Generate version file
function(GenerateVersionFile)
    file(WRITE ${CMAKE_SOURCE_DIR}/Resources/Assets/Generated/VersionInfo "VERSION=${PROJECT_VERSION}\nSERVER=${CRL_SERVER_IP}\nPROTOCOL=${CRL_SERVER_PROTOCOL}\nDESTINATION=${CRL_SERVER_DESTINATION}\nDESTINATION_VERSIONINFO=${CRL_SERVER_VERSIONINFO_DESTINATION}\nLOG_LEVEL=${VIEWER_LOG_LEVEL}")
endfunction()

# Set install directory
if (UNIX)
    set(INSTALL_DIRECTORY "${CMAKE_BINARY_DIR}/multisense_${PROJECT_VERSION}_${CMAKE_SYSTEM_PROCESSOR}/opt/multisense")
elseif (WIN32)
    set(INSTALL_DIRECTORY "${CMAKE_BINARY_DIR}/multisense_${PROJECT_VERSION}_${CMAKE_SYSTEM_PROCESSOR}/")
endif ()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "[VkRenderINFO]: Set install directory to ${INSTALL_DIRECTORY}")
    set(CMAKE_INSTALL_PREFIX "${INSTALL_DIRECTORY}" CACHE PATH "default install path" FORCE)
endif ()