include(FetchContent)
FetchContent_Declare(
    matplotlibcpp
    GIT_REPOSITORY https://github.com/DamynChipman/matplotlib-cpp
    GIT_TAG 73d633f # Release v0.1
)

FetchContent_GetProperties(matplotlibcpp)
FetchContent_Populate(matplotlibcpp)

if(PYTHON_ENV_PATH)

    if(PYTHON_VERSION)

        set(PYTHON_INCLUDE_DIR ${PYTHON_ENV_PATH}/include/${PYTHON_VERSION} CACHE INTERNAL "Python include directory with Python.h")
        set(PYTHON_LIBRARY_DIR ${PYTHON_ENV_PATH}/lib CACHE INTERNAL "Python library directory")
        set(NUMPY_INCLUDE_DIR ${PYTHON_ENV_PATH}/lib/${PYTHON_VERSION}/site-packages/numpy/core/include CACHE INTERNAL "Numpy header file location")

    else()

        message(FATAL_ERROR "PYTHON_VERSION not set in configuration! See docs for installation steps.")

    endif()


else()

    message(FATAL_ERROR "PYTHON_ENV_PATH not set in configuration! See docs for installation steps.")

endif()