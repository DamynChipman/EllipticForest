add_subdirectory(hello)
add_subdirectory(poisson)
add_subdirectory(toybox)

if(WITH_FORESTCLAW)
    add_subdirectory(ForestClaw)
endif()