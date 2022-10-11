if(P4EST_PATH)

    if(EXISTS ${P4EST_PATH}/include/p4est.h)
        set(P4EST_INCLUDE ${P4EST_PATH}/include CACHE INTERNAL "p4est include directory")
    else()
        message(FATAL_ERROR "Could not find p4est.h")
    endif()

    if(EXISTS ${P4EST_PATH}/lib/libp4est.dylib)
        set(P4EST_LIB ${P4EST_PATH}/lib CACHE INTERNAL "p4est library directory")
    elseif(EXISTS ${P4EST_PATH}/lib/libp4est.a)
        set(P4EST_LIB ${P4EST_PATH}/lib CACHE INTERNAL "p4est library directory")
    elseif(EXISTS ${P4EST_PATH}/lib/libp4est.so)
        set(P4EST_LIB ${P4EST_PATH}/lib CACHE INTERNAL "p4est library directory")
    else()
        message(FATAL_ERROR "Could not find libp4est")
    endif()

    set(P4EST_FOUND TRUE CACHE INTERNAL "Flag for if p4est was found")

    message(STATUS "Found external p4est:")
    message(STATUS "  P4EST_INCLUDE = " ${P4EST_INCLUDE})
    message(STATUS "  P4EST_LIB = " ${P4EST_LIB})

else()

    message(FATAL_ERROR "P4EST_PATH was not set in configuration!")
    
endif()