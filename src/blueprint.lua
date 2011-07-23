-- location of the gpu computing sdk (for common includes, libraries, and such)
-- this is the BASE PATH to the sdk, in other words the directory where
-- the C, doc, OpenCL, and shared directories live
gpu_sdk_path "~/NVIDIA_GPU_Computing_SDK"

-- set our default config, the one that's run if we don't specify one on the
-- command line
default "debug"

-- define a debug config, where debugging options are turned on, profiling
-- options are turned off, and optimizing is turned off
config "debug"
    -- output "build-debug"
    output "../terlR"
    debugging "on"
    defines {"DEBUG"}
    libdirs {"../lib/pngwriter/lib", "../lib", "/home/cbrenton/NVIDIA_GPU_Computing_SDK/C/lib"}
    includes {"./", "../lib", "../lib/pngwriter/include", "/home/cbrenton/NVIDIA_GPU_Computing_SDK/C/common/inc/"}
    cflags {"-Wall -DNO_FREETYPE"}
    libs {"m", "png", "z", "pngwriter", "cudart"}
    compute_capability "2.0"

-- same as above, but define a "profile" config
config "profile"
    -- output "build-profile"
    output "../terlR"
    debugging "on"
    profiling "on"
    libdirs {"../lib/pngwriter/lib", "../lib", "/home/cbrenton/NVIDIA_GPU_Computing_SDK/C/lib"}
    includes {"./", "../lib", "../lib/pngwriter/include", "/home/cbrenton/NVIDIA_GPU_Computing_SDK/C/common/inc/"}
    cflags {"-Wall -DNO_FREETYPE"}
    libs {"m", "png", "z", "pngwriter", "cudart"}
    compute_capability "2.0"

-- same as above, but define a "release" config
config "release"
    -- output "build-release"
    output "../terlR"
    optimizing "on"
    defines {"RELEASE"}
    libdirs {"../lib/pngwriter/lib", "../lib", "/home/cbrenton/NVIDIA_GPU_Computing_SDK/C/lib"}
    includes {"./", "../lib", "../lib/pngwriter/include", "/home/cbrenton/NVIDIA_GPU_Computing_SDK/C/common/inc/"}
    cflags {"-Wall -DNO_FREETYPE"}
    libs {"m", "png", "z", "pngwriter", "cudart"}
    compute_capability "2.0"
