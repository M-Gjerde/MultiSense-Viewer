cmake_minimum_required(VERSION 3.20)
project(
        MultiSense-Viewer
        DESCRIPTION "A MultiSense Viewer Application"
        LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set application version. Remember that it should always match the pushed tag in the form of v1.*-*
set(VERSION_MAJOR "1")
set(VERSION_MINOR "1")
set(VERSION_PATCH "7")
set(ARCHITECTURE "amd64")

# Set install directory. Used to generate installers for both Windows and Ubuntu using their native approaches
set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/)

option(BUILD_DOC "Build documentation" ON)
# Set a default build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE
            Release
            CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui, ccmake
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Find Vulkan
find_package(Vulkan REQUIRED)

if (WIN32)
    add_executable(${PROJECT_NAME} WIN32)
else ()
    add_executable(${PROJECT_NAME})
endif ()

include(cmake/Common.cmake)
include(cmake/CompilerWarnings.cmake)

find_package(Doxygen)
if (Doxygen_FOUND)
    add_subdirectory(docs)
else ()
    message(STATUS "Doxygen not found, not building docs")
endif ()


include_directories(include)
include_directories(Assets)
include_directories(SYSTEM external/ffmpeg/include)
include_directories(SYSTEM ${Vulkan_INCLUDE_DIRS})
ExportScriptIncludes()
GenerateVersionFile()

set(LOGGER_SRC
        src/Tools/Logger.cpp
)

set(ENGINE_SRC
        src/Core/Buffer.cpp
        src/Core/Texture.cpp
        src/Core/VulkanDevice.cpp
        src/Core/VulkanRenderer.cpp
        src/Core/VulkanSwapchain.cpp
        src/Core/RenderResource.cpp
        src/CRLCamera/CRLPhysicalCamera.cpp
        src/ImGui/GuiManager.cpp
        src/ModelLoaders/CRLCameraModels.cpp
        src/ModelLoaders/CustomModels.cpp
        src/ModelLoaders/GLTFModel.cpp

        src/Renderer/Renderer.cpp
        src/Tools/StbInclude.cpp
        src/CRLCamera/CameraConnection.cpp
        src/ImGui/Widgets.cpp
        src/ImGui/LayerFactory.cpp
        src/Renderer/UsageMonitor.cpp
        src/Core/RendererConfig.cpp
        src/Core/ServerConnection.cpp
        src/Core/CommandBuffer.cpp
        src/ModelLoaders/ImageView.cpp
)

set(SCRIPTS_LIB_SRC
        include/Viewer/Scripts/MultiSenseRenderer/MultiSenseCamera.cpp
        include/Viewer/Scripts/MultiSenseRenderer/Example.cpp
        include/Viewer/Scripts/MultiSenseRenderer/Pointcloud/PointCloud.cpp
        include/Viewer/Scripts/MultiSenseRenderer/SceneGizmos/Gizmos.cpp
        include/Viewer/Scripts/MultiSenseRenderer/Video/RecordFrames.cpp
        include/Viewer/Scripts/MultiSenseRenderer/Video/Previews/Single/SingleLayout.cpp
        include/Viewer/Scripts/MultiSenseRenderer/Video/Previews/Double/DoubleTop.cpp
        include/Viewer/Scripts/MultiSenseRenderer/Video/Previews/Double/DoubleBot.cpp
        include/Viewer/Scripts/MultiSenseRenderer/Video/Previews/Quad/One.cpp
        include/Viewer/Scripts/MultiSenseRenderer/Video/Previews/Quad/Two.cpp
        include/Viewer/Scripts/MultiSenseRenderer/Video/Previews/Quad/Three.cpp
        include/Viewer/Scripts/MultiSenseRenderer/Video/Previews/Quad/Four.cpp
        include/Viewer/Scripts/MultiSenseRenderer/SceneGizmos/Skybox.cpp

        include/Viewer/Scripts/Private/ScriptUtils.h
        include/Viewer/Scripts/Private/RecordUtilities.h

        include/Viewer/Scripts/Renderer3D/Grid.cpp
        include/Viewer/Scripts/Renderer3D/MultiSense.cpp
        include/Viewer/Scripts/Renderer3D/Example/Example3D.cpp
        include/Viewer/Scripts/Renderer3D/ImageViewer.cpp
)

set(LIBRARIES "Vulkan::Vulkan;glfw;SimpleIni;tiff;MultiSense;LibAutoConnect;ktx;nlohmann_json::nlohmann_json;httplib::httplib;rosbag_cpp_writer;")

# Copy Assets folders to output directory
file(GLOB ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/Assets/*")
file(COPY ${ASSETS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Assets)

if (UNIX) ## Linux
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message("   INFO: Default Install prefix was chosen so set install path to ${INSTALL_DIRECTORY}")
        set(CMAKE_INSTALL_PREFIX "${INSTALL_DIRECTORY}"
                CACHE PATH "default install path" FORCE)
    endif ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")

    add_library(libsll STATIC IMPORTED GLOBAL)
    set_target_properties(libsll PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/lib/libssl.a")

    add_library(libcrypto STATIC IMPORTED GLOBAL)
    set_target_properties(libcrypto PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/lib/libcrypto.a")

    # GTK for native file dialog
    # Locate pkg-config
    find_package(PkgConfig REQUIRED)
    # Check for GTK+ using pkg-config
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
    # Add GTK+ include directories
    include_directories(${GTK3_INCLUDE_DIRS})
    # Add GTK+ linking directories
    link_directories(${GTK3_STATIC_LIBRARY_DIRS})
    # Add GTK+ compiler flags
    add_definitions(${GTK3_CFLAGS_OTHER})

    # Link all source files directly to the main project target
    target_sources(${PROJECT_NAME} PRIVATE
            ${LOGGER_SRC}
            ${ENGINE_SRC}
            ${SCRIPTS_LIB_SRC}
            ${IMGUI_SOURCES}
            src/main.cpp
    )

    foreach(IMGUI_FILE ${IMGUI_SOURCES})
        set_source_files_properties(${IMGUI_FILE} PROPERTIES COMPILE_FLAGS "-w")
    endforeach()

    # Direct linking of libraries
    target_link_libraries(${PROJECT_NAME} PRIVATE
            fmt::fmt
            ${LIBRARIES}
            ${GTK3_STATIC_LIBRARIES}
            libsll
            libcrypto
            -lavutil
            -lswscale
            MultiSense
            httplib::httplib
            rosbag_cpp_writer
    )
    # Add dependencies directly to the project
    add_dependencies(${PROJECT_NAME}
            AutoConnect
            rosbag_cpp_writer
    )
    set_project_warnings(${PROJECT_NAME})


    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/AutoConnect CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/AutoConnectLauncher.sh CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})

    add_custom_command(TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/internal/AutoConnect/AutoConnect ${CMAKE_BINARY_DIR}/
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/Assets/Tools/Ubuntu/AutoConnectLauncher.sh ${CMAKE_BINARY_DIR}/
    )
endif ()

if (WIN32)
    # Set install directory if not specified
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message("   INFO: Default Install prefix was chosen so set install path to ${INSTALL_DIRECTORY}")
        set(CMAKE_INSTALL_PREFIX "${INSTALL_DIRECTORY}"
                CACHE PATH "default install path" FORCE)
    endif ()

    if (MSVC)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
        add_definitions(-DNOMINMAX)
    endif ()

    # Include directory for WinPcap
    include_directories(internal/AutoConnect/include/WinPcap)

    add_library(avutil STATIC IMPORTED GLOBAL)
    set_target_properties(avutil PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/avutil.lib")

    add_library(swscale STATIC IMPORTED GLOBAL)
    set_target_properties(swscale PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/swscale.lib")

    add_library(libsll STATIC IMPORTED GLOBAL)
    set_target_properties(libsll PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/lib/libssl.lib")

    add_library(libcrypto STATIC IMPORTED GLOBAL)
    set_target_properties(libcrypto PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/lib/libcrypto.lib")

    add_library(CustomLoggerInterface INTERFACE)
    target_link_libraries(CustomLoggerInterface INTERFACE fmt::fmt)

    add_library(CustomLogger ${LOGGER_SRC})
    target_link_libraries(CustomLogger PRIVATE CustomLoggerInterface)

    add_library(VulkanRenderer STATIC ${ENGINE_SRC})
    target_link_libraries(VulkanRenderer ${LIBRARIES} CustomLogger libsll libcrypto)

    add_library(ScriptLibrary STATIC ${SCRIPTS_LIB_SRC})
    target_link_libraries(ScriptLibrary VulkanRenderer avutil swscale libsll libcrypto rosbag_cpp_writer)

    target_sources(${PROJECT_NAME} PRIVATE src/main.cpp)

    target_link_libraries(${PROJECT_NAME} VulkanRenderer ScriptLibrary CustomLoggerInterface)
    add_dependencies(${PROJECT_NAME} AutoConnect)

    set_project_warnings(ScriptLibrary)
    set_project_warnings(VulkanRenderer)
    set_project_warnings(${PROJECT_NAME})

    #AutoConnect
    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
    set_target_properties(AutoConnect PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(AutoConnect PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
    set_target_properties(AutoConnect PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
    set_target_properties(AutoConnect PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(AutoConnect PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
    set_target_properties(AutoConnect PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
    set_target_properties(MultiSense PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(MultiSense PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
    set_target_properties(MultiSense PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

    # create a list of third part DLLs to copy
    set(THIRD_PARTY_DLLS
            #FFMPEG
            ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/swscale-6.dll
            ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/avutil-57.dll
            #OpenSSL
            ${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/bin/libcrypto-1_1-x64.dll
            ${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/bin/libssl-1_1-x64.dll
            # WinPcap DLLs
            ${CMAKE_CURRENT_SOURCE_DIR}/internal/AutoConnect/WinPcap/Packet.dll
            ${CMAKE_CURRENT_SOURCE_DIR}/internal/AutoConnect/WinPcap/wpcap.dll
    )
    # do the copying
    foreach (file_i ${THIRD_PARTY_DLLS})
        add_custom_command(
                TARGET VulkanRenderer
                POST_BUILD
                COMMAND ${CMAKE_COMMAND}
                ARGS -E copy ${file_i} ${CMAKE_BINARY_DIR}/)
    endforeach (file_i)

    ### INSTALL TARGETS ###
    # VkRender Viewer DLL and libraries
    install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release LIBRARY DESTINATION ${INSTALL_DIRECTORY})
    # LibMultiSense and WinPCap dll
    install(FILES
            ${CMAKE_BINARY_DIR}/MultiSense.dll
            ${CMAKE_BINARY_DIR}/Packet.dll
            ${CMAKE_BINARY_DIR}/wpcap.dll
            ${CMAKE_BINARY_DIR}/swscale-6.dll
            ${CMAKE_BINARY_DIR}/avutil-57.dll
            ${CMAKE_BINARY_DIR}/libcrypto-1_1-x64.dll
            ${CMAKE_BINARY_DIR}/libssl-1_1-x64.dll
            CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})

    # AutoConnect
    install(PROGRAMS ${CMAKE_BINARY_DIR}/AutoConnect.exe CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})
    # Copy Assets
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})

    if (MSVC)
        if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
            message("\n\t[ WARNING ]\n\n\tCMake version lower than 3.6.\n\n\t - Please update CMake and rerun; OR\n\t - Manually set 'GLFW-CMake-starter' as StartUp Project in Visual Studio.\n")
        else ()
            set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
            set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
        endif ()
    endif ()
endif ()

# Compile definitions
if (CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_definitions(${PROJECT_NAME} PRIVATE MULTISENSE_VIEWER_PRODUCTION)
else ()
    target_compile_definitions(${PROJECT_NAME} PRIVATE MULTISENSE_VIEWER_DEBUG)
endif ()
