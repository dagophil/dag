add_executable(dagraph_test
    dagraph_test.cxx
)

target_link_libraries(dagraph_test
    ${VIGRA_IMPEX_LIBRARY}
    ${HDF5_LIBRARIES}
)

