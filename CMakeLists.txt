cmake_minimum_required(VERSION 3.20)

# Project definition
project(MultiSense-Viewer
        VERSION 1.2.0
        DESCRIPTION "A MultiSense Viewer Application"
        LANGUAGES C CXX
)

# Options
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" FALSE)
option(ENABLE_PYTORCH "Compile with PyTorch" FALSE)
option(BUILD_DOC "Build documentation" ON)
option(PROPRIETARY_GIGEVISION_MODULE "Include the proprietary GigE-Vision module" OFF)

option(ENABLE_SYCL "Compile with SYCL support. Provided that llvm-sycl clang++ is in path" FALSE)
set(SYCL_IMPLEMENTATION "adaptivecpp" CACHE STRING "Select SYCL backend (dpcpp or adaptivecpp)")
option(SYCL_BACKEND "Choose the SYCL backend (nvidia or amd)" "nvidia")
set_property(CACHE SYCL_IMPLEMENTATION PROPERTY STRINGS "dpcpp" "adaptivecpp")
option(OIDN_DIR "Set the path for OpenImageDenoise" OFF)


# Paths for proprietary module
set(GIGEVISION_LIB_PATH "" CACHE PATH "Path to the proprietary GigE-Vision module library")
set(GIGEVISION_HEADERS_PATH "" CACHE PATH "Path to the proprietary GigE-Vision module headers")

# Compiler settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME})

# Set default build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "[VkRenderINFO]: Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
    message(STATUS "[VkRenderINFO]: Build type is ${CMAKE_BUILD_TYPE}")

endif ()

# Load Dependencies
include(cmake/Common.cmake)
include(cmake/CompilerWarnings.cmake)


# Check if PyTorch is available with ROCm or CUDA
set(PYTORCH_AVAILABLE FALSE)
set(USE_CUDA FALSE)
set(USE_ROCM FALSE)

if (ENABLE_PYTORCH)

    message(STATUS "[VkRenderINFO]: Attempting to find LibTorch")
    set(hipblas-common_DIR /opt/rocm-6.3.1/lib/cmake/hipblas-common)
    set(CMAKE_CUDA_ARCHITECTURES 89)

    find_package(Torch REQUIRED HINTS
            /home/magnus/libraries/libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124/libtorch
            /home/magnus-desktop/libraries/libtorch-cxx11-abi-shared-with-deps-2.5.1+rocm6.2/libtorch

    )

    if (Torch_FOUND)
        message(STATUS "[VkRenderINFO]: PyTorch found at ${TORCH_INSTALL_PREFIX}")
        # Check if PyTorch is built for CUDA
        if (DEFINED TORCH_CUDA_LIBRARIES)
            set(PYTORCH_AVAILABLE TRUE)
            set(USE_CUDA TRUE)
            message(STATUS "[VkRenderINFO]: PyTorch is configured with CUDA.")
        endif ()

        # Check if PyTorch is built for ROCm
        if (DEFINED TORCH_HIP_VERSION)
            set(PYTORCH_AVAILABLE TRUE)
            set(USE_ROCM TRUE)
            message(STATUS "[VkRenderINFO]: PyTorch is configured with ROCm.")
        endif ()

        if (NOT USE_CUDA AND NOT USE_ROCM)
            message(FATAL_ERROR "PyTorch is not configured with CUDA or ROCm. Skipping PyTorch integration.")
        endif ()
    else ()
        message(FATAL_ERROR "PyTorch not found. Skipping PyTorch integration.")
    endif ()
endif ()

if (PYTORCH_AVAILABLE)
    if (USE_CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)


        message(STATUS "[VkRenderINFO]: CUDA language support enabled.")
        target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} cuda cudart cupti)
        target_link_directories(${PROJECT_NAME} PRIVATE /usr/local/cuda/lib64)

    elseif (USE_ROCM)
        message(STATUS "[VkRenderINFO]: HIP language support enabled.")
        set(CMAKE_HIP_STANDARD 20)
        set(CMAKE_HIP_STANDARD_REQUIRED ON)
        set(CMAKE_HIP_EXTENSIONS OFF)
        # Add any ROCm-specific setup if necessary (e.g., include ROCm libraries or headers)
        message(STATUS "[VkRenderINFO]: ROCm-specific setup would go here if needed.")
        #target_link_directories(${PROJECT_NAME} PRIVATE /opt/amdgpu/lib/x86_64-linux-gnu/)

    endif ()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    message(STATUS "[VkRenderINFO]: PyTorch libraries successfully linked.")
    target_compile_definitions(${PROJECT_NAME} PRIVATE DIFF_RENDERER_ENABLED)

else ()
    message(STATUS "[VkRenderINFO]: Skipping PyTorch setup as it is not available.")
endif ()


# SYCL Backend selectionDefault
if (ENABLE_SYCL)
    message(STATUS "SYCL backend: ${SYCL_IMPLEMENTATION}")

    if(SYCL_BACKEND STREQUAL "nvidia")
        set(SYCL_TARGET "spir64,nvptx64-nvidia-cuda")
        message(STATUS "[VkRenderINFO]: Using NVIDIA backend for SYCL")
    elseif(SYCL_BACKEND STREQUAL "amd")
        set(SYCL_TARGET "spir64,amdgcn-amd-amdhsa")
        set(SYCL_TARGET_OPTIONS "-Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1100")
        message(STATUS "[VkRenderINFO]: Using AMD backend for SYCL")
    else()
        message(FATAL_ERROR "Invalid SYCL_BACKEND option. Choose 'nvidia' or 'amd'.")
    endif()

    if (SYCL_IMPLEMENTATION STREQUAL "dpcpp")
        set(CMAKE_CXX_COMPILER "/home/magnus/sycl_workspace/llvm/build/bin/clang++")

        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O3 -fsycl-targets=${SYCL_TARGET} ${SYCL_TARGET_OPTIONS}")
            message(STATUS "[VkRenderINFO]: Compiling using DPCPP with Optimization")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -g -O0 -fsycl-targets=${SYCL_TARGET} ${SYCL_TARGET_OPTIONS}")
            message(STATUS "[VkRenderINFO]: Compiling using DPCPP with Debug flags")
        endif()

        message(STATUS "[VkRenderINFO]: SYCL DPC++ support enabled. Compiler: ${CMAKE_CXX_COMPILER} Flags: ${CMAKE_CXX_FLAGS}")

    elseif (SYCL_IMPLEMENTATION STREQUAL "adaptivecpp")
        # Find the AdaptiveCpp package
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            set(ACPP_TARGETS "generic")
        else()
            set(ACPP_TARGETS "omp")
        endif()
        target_link_options(${PROJECT_NAME} PRIVATE -L/usr/lib/llvm-14/lib)
        target_compile_options(${PROJECT_NAME} PRIVATE -I/usr/lib/llvm-14/lib/clang/14.0.0/include)

        find_package(AdaptiveCpp REQUIRED)
        # Check if ACPP_TARGETS is passed via command line or environment
        # Print a message showing which targets are being used
        message(STATUS "[VkRenderINFO]: Using AdaptiveCpp with targets: ${ACPP_TARGETS} to project name ${PROJECT_NAME}")
        # Add sycl to target using AdaptiveCpp's helper
        add_sycl_to_target(TARGET ${PROJECT_NAME} TARGETS ${ACPP_TARGETS})
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_options(${PROJECT_NAME} PRIVATE -O0 -g )
            #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
            message(STATUS "[VkRenderINFO]: Compiling using AdaptiveCpp without optimization")
        endif()
    else ()
        message(FATAL_ERROR "[VkRenderERROR]: Unknown SYCL backend. Choose either 'dpcpp' or 'adaptivecpp'.")
    endif ()
    target_compile_definitions(${PROJECT_NAME} PRIVATE SYCL_ENABLED)
    message(STATUS "[VkRenderINFO] Added definition: SYCL_ENABLED")

    # Include the OIDN headers
    include_directories("${OIDN_DIR}/include")
    # Link the OIDN library
    target_link_libraries(${PROJECT_NAME} PRIVATE
            "${OIDN_DIR}/lib/libOpenImageDenoise.so"
    )
    message(STATUS "[VkRenderINFO]: OIDN Include Dir: ${OIDN_DIR}/include")
    message(STATUS "[VkRenderINFO]: OIDN Lib Dir: ${OIDN_DIR}/lib")

else ()
    message(STATUS "[VkRenderINFO]: SYCL support not enabled.")
endif ()


# Add version file generation
GenerateVersionFile()

# Find Vulkan
find_package(Vulkan REQUIRED)

# Include directories
include_directories(src)
include_directories(Resources/Assets)
include_directories(SYSTEM external/ffmpeg/include)
include_directories(SYSTEM ${Vulkan_INCLUDE_DIRS})
include_directories(SYSTEM src/Viewer/Rendering/ImGui/Custom)

# Find Doxygen for documentation
find_package(Doxygen)
if (Doxygen_FOUND AND BUILD_DOC)
    add_subdirectory(docs)
else ()
    message(STATUS "[VkRenderINFO]: Doxygen not found, not building docs")
endif ()

# Set application version
set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${PROJECT_VERSION}_${CMAKE_SYSTEM_PROCESSOR}/)

set(SOURCE_FILES
        src/main.cpp
        src/Viewer/Rendering/Core/Buffer.cpp
        src/Viewer/Rendering/Core/VulkanDevice.cpp
        src/Viewer/Rendering/Core/VulkanRenderer.cpp
        src/Viewer/Rendering/Core/VulkanSwapchain.cpp
        src/Viewer/Rendering/Core/UUID.cpp
        src/Viewer/Rendering/Core/ServerConnection.cpp
        src/Viewer/Rendering/Core/CommandBuffer.cpp
        src/Viewer/Rendering/Core/VulkanRenderPass.cpp
        src/Viewer/Rendering/Core/VulkanResourceManager.cpp
        src/Viewer/Rendering/Core/VulkanGraphicsPipeline.cpp
        src/Viewer/Rendering/Core/VulkanImage.cpp
        src/Viewer/Rendering/Core/VulkanFramebuffer.cpp
        src/Viewer/Rendering/Core/VulkanTexture.cpp
        src/Viewer/Rendering/Core/VulkanTexture.h
        src/Viewer/Scenes/Entity.cpp

        src/Viewer/Application/ApplicationConfig.cpp
        src/Viewer/Application/pch.cpp
        src/Viewer/Application/Application.cpp
        src/Viewer/Application/UsageMonitor.cpp
        src/Viewer/Application/ApplicationUserSettings.h
        src/Viewer/Application/ProjectSerializer.cpp

        src/Viewer/Rendering/RenderResources/DefaultGraphicsPipeline.cpp
        src/Viewer/Rendering/RenderResources/GraphicsPipeline2D.cpp

        src/Viewer/Rendering/MeshData.cpp
        src/Viewer/Rendering/VulkanMeshResourceManager.cpp
        src/Viewer/Rendering/MeshManager.cpp
        src/Viewer/Rendering/IMeshParameters.cpp

        src/Viewer/Scenes/Scene.cpp
        src/Viewer/Scenes/SceneSerializer.cpp

        src/Viewer/Rendering/Components/ImageComponent.cpp
        src/Viewer/Rendering/Components/PointCloudComponent.h
        src/Viewer/Rendering/Components/GaussianComponent.cpp

        src/Viewer/Rendering/ImGui/Widgets.cpp
        src/Viewer/Rendering/ImGui/GuiManager.cpp
        src/Viewer/Rendering/ImGui/AdditionalWindows/DebugWindow.h
        src/Viewer/Rendering/ImGui/LayerFactory.cpp
        src/Viewer/Rendering/ImGui/GuiResources.cpp

        src/Viewer/Rendering/Editors/Editor.cpp
        src/Viewer/Rendering/Editors/3DViewport/Editor3DViewport.cpp
        src/Viewer/Rendering/Editors/ImageEditor/EditorImage.cpp
        src/Viewer/Rendering/Editors/MainContextLayer.h
        src/Viewer/Rendering/Editors/MultiSenseViewer/Layers/LayoutSettingsLayer.h
        src/Viewer/Rendering/Editors/Video/VideoSource.cpp
        src/Viewer/Rendering/Editors/Video/VideoPlaybackSystem.cpp
        src/Viewer/Rendering/Editors/SceneRenderer.cpp
        src/Viewer/Rendering/Editors/SceneHierarchy/SceneHierarchyLayer.cpp

        src/Viewer/Rendering/Editors/PathTracer/EditorPathTracer.cpp

        src/Viewer/Rendering/Core/PipelineKey.cpp
        src/Viewer/Rendering/Editors/Properties/PropertiesLayer.cpp
        src/Viewer/Rendering/Core/PipelineManager.cpp
        src/Viewer/Rendering/Core/DescriptorSetManager.cpp

        src/Viewer/Tools/StbInclude.cpp
        src/Viewer/Tools/Logger.cpp
        src/Viewer/Tools/ThreadPool.h

        src/Viewer/Modules/MultiSense/MultiSenseRendererBridge.cpp
        src/Viewer/Modules/MultiSense/MultiSenseDevice.cpp
        src/Viewer/Modules/MultiSense/CommonHeader.h
        src/Viewer/Modules/MultiSense/MultiSenseTaskManager.cpp
        src/Viewer/Modules/MultiSense/LibMultiSense/LibMultiSenseConnector.cpp
        src/Viewer/Application/ApplicationUserSettingsSerializer.cpp
        src/Viewer/Application/ApplicationUserSettingsSerializer.h
)

# Libraries
set(LIBRARIES
        Vulkan::Vulkan
        glfw
        imgui
        tiff
        nlohmann_json::nlohmann_json
        httplib::httplib
        tinyobjloader
        GPUOpen::VulkanMemoryAllocator
        tinyply
        MultiSense
        yaml-cpp
        fmt::fmt
)

# Add SYCL-specific files
if (ENABLE_SYCL)
    set(SYCL_SOURCES
            #src/Viewer/Rendering/RenderResources/2DGS/SYCLGaussian2D.cpp
            #src/Viewer/Rendering/RenderResources/3DGS/SYCLGaussian3D.cpp
            src/Viewer/Rendering/Editors/GaussianViewer/EditorGaussianViewer.cpp
            src/Viewer/Rendering/RenderResources/PathTracer/PathTracer.cpp
            src/Viewer/Tools/SyclDeviceSelector.h
    )
    list(APPEND SOURCE_FILES ${SYCL_SOURCES})
    message(STATUS "[VkRenderINFO]: SYCL: Adding Gaussian Renderer to source")
else()
    message(STATUS "[VkRenderINFO]: Skipped Adding SYCL Sources")
endif ()# Add SYCL-specific files

if (${ENABLE_PYTORCH})
    set (DIFF_RENDER_SOURCES
            src/Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildModule.cpp
            src/Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildFunction.cpp
            src/Viewer/Rendering/Editors/DifferentiableEditor/EditorDifferentiableRenderer.cpp
    )
    list(APPEND SOURCE_FILES ${DIFF_RENDER_SOURCES})

    message(STATUS "[VkRenderINFO]: SYCL: Adding Pytorch differential Renderer to source")
endif()

# Add proprietary GigE-Vision module
if (PROPRIETARY_GIGEVISION_MODULE)
    list(APPEND SOURCE_FILES src/Viewer/Modules/MultiSense/GigE-Vision/GigEVisionConnector.cpp)
    list(APPEND LIBRARIES ${GIGEVISION_LIB})
    target_compile_definitions(${PROJECT_NAME} PRIVATE VKRENDER_GIGEVISION_ENABLED)
endif ()

#Custom command and target to copy assets if they have changed
file(GLOB ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/Resources/Assets/*")
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Resources/Assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/Resources/Assets ${CMAKE_CURRENT_BINARY_DIR}/Resources/Assets
        DEPENDS ${ASSETS}
        COMMENT "Copying assets..."
)
add_custom_target(copy_assets ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Resources/Assets)

# Build configurations for different platforms
if (UNIX)
    # Linux-specific settings
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message(STATUS "INFO: Default Install prefix was chosen so set install path to ${INSTALL_DIRECTORY}")
        set(CMAKE_INSTALL_PREFIX "${INSTALL_DIRECTORY}" CACHE PATH "default install path" FORCE)
    endif ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")

    # GTK for native file dialog
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
    include_directories(${GTK3_INCLUDE_DIRS})
    link_directories(${GTK3_LIBRARY_DIRS})
    add_definitions(${GTK3_CFLAGS_OTHER})

    # Build executable
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES} ${GTK3_LIBRARIES} fmt::fmt)
    add_dependencies(${PROJECT_NAME} copy_assets)

    # Set warnings
    set_project_warnings(${PROJECT_NAME})

    # Install configuration
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/Resources/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})

elseif (WIN32)
    # Windows-specific settings
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message(STATUS "INFO: Default Install prefix was chosen so set install path to ${INSTALL_DIRECTORY}")
        set(CMAKE_INSTALL_PREFIX "${INSTALL_DIRECTORY}" CACHE PATH "default install path" FORCE)
    endif ()

    if (MSVC)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
        add_definitions(-DNOMINMAX)
    endif ()

    # Build executable
    target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})
    target_link_libraries(${PROJECT_NAME} ${LIBRARIES} fmt::fmt userenv.lib)
    add_dependencies(${PROJECT_NAME} copy_assets)

    # Set warnings
    set_project_warnings(${PROJECT_NAME})

    # Set runtime output directory
    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
    set_target_properties(MultiSense PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

    # Install configuration
    install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release LIBRARY DESTINATION ${INSTALL_DIRECTORY})
    install(FILES ${CMAKE_BINARY_DIR}/MultiSense.dll CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})
    install(DIRECTORY Resources/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})

endif ()

# Compile definitions for different build types
if (CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_definitions(${PROJECT_NAME} PRIVATE VKRENDER_MULTISENSE_VIEWER_PRODUCTION)
else ()
    target_compile_definitions(${PROJECT_NAME} PRIVATE VKRENDER_MULTISENSE_VIEWER_DEBUG)
endif ()
