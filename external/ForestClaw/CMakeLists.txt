if(FORESTCLAW_PATH)

    if(EXISTS ${FORESTCLAW_PATH}/include/forestclaw2d.h)
        set(FORESTCLAW_INCLUDE ${FORESTCLAW_PATH}/include CACHE INTERNAL "ForestClaw include directory")
    else()
        message(FATAL_ERROR "Could not find forestclaw2d.h")
    endif()

    if(EXISTS ${FORESTCLAW_PATH}/lib/libforestclaw.a)
        set(FORESTCLAW_LIB ${FORESTCLAW_PATH}/lib CACHE INTERNAL "ForestClaw library directory")
    else()
        message(FATAL_ERROR "Could not find libforestclaw")
    endif()

    set(FORESTCLAW_FOUND TRUE CACHE INTERNAL "Flag for if ForestClaw was found")

    message(STATUS "Found external ForestClaw:")
    message(STATUS "  FORESTCLAW_INCLUDE = " ${FORESTCLAW_INCLUDE})
    message(STATUS "  FORESTCLAW_LIB = " ${FORESTCLAW_LIB})

else()

    message(FATAL_ERROR "FORESTCLAW_PATH was not set in configuration!")

endif()