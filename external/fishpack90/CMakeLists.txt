set(FISHPACK_SOURCES 
	${CMAKE_CURRENT_SOURCE_DIR}/src/fish.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/blktri.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/cblktri.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/cmgnbn.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/comf.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/fftpack.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/genbun.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/gnbnaux.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/hstcrt.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/hstcsp.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/hstcyl.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/hstplr.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/hstssp.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/hw3crt.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/hwscrt.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/hwscsp.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/hwscyl.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/hwsplr.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/hwsssp.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/pois3d.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/poistg.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/sepaux.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/sepeli.f
	${CMAKE_CURRENT_SOURCE_DIR}/src/sepx4.f
)

# set(FISHPACK_TESTS
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/tblktri.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/tcblktri.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/tcmgnbn.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/tgenbun.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/thstcrt.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/thstcsp.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/thstcyl.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/thstplr.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/thstssp.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/thw3crt.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/thwscrt.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/thwscsp.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/thwscyl.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/thwsplr.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/thwsssp.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/tpois3d.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/tpoistg.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/tsepeli.f
# 	${CMAKE_CURRENT_SOURCE_DIR}/test/tsepx4.f
# )

# set(CMAKE_Fortran_FLAGS "-Wno-unused-dummy-argument -fallow-argument-mismatch -fdefault-real-8")

add_library(
	fishpack90 STATIC
	${FISHPACK_SOURCES}
)

target_compile_options(
	fishpack90 BEFORE PRIVATE 
	"-w" "-fallow-argument-mismatch" "-fdefault-real-8"
)

# set(FISHPACK90_LIB fishpack90)
set(FISHPACK90_FOUND TRUE CACHE INTERNAL "Flag for FISHPACK90 ready")
set(FISHPACK90_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE INTERNAL "FISHPACK90 include directory")
set(FISHPACK90_LIB ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "FISHPACK90 library directory")

# install(
# 	TARGETS fishpack90
# 	EXPORT ${PROJECT_NAME}Targets
# )

# foreach(TEST_FILE ${FISHPACK_TESTS})
# 	set(TEST_NAME)
# 	get_filename_component(TEST_NAME ${TEST_FILE} NAME)
# 	add_custom_command(
# 		TARGET fishpack90 POST_BUILD
# 		WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
# 		COMMAND gfortran -fdefault-real-8 -fallow-argument-mismatch -L${CMAKE_CURRENT_BINARY_DIR} -lfishpack90 ${TEST_FILE}
# 		COMMENT "Running FISHPACK tests for ${TEST_NAME}"
# 	)
# 	add_custom_command(
# 		TARGET fishpack90 POST_BUILD
# 		WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
# 		COMMAND ${CMAKE_CURRENT_BINARY_DIR}/a.out
# 	)
# 	add_custom_command(
# 		TARGET fishpack90 POST_BUILD
# 		WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
# 		COMMAND rm -f ${CMAKE_CURRENT_BINARY_DIR}/a.out
# 	)
# endforeach()