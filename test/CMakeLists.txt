add_executable(dagraph_test
    dagraph_test.cxx
)

add_executable(randomforest_test
    randomforest_test.cxx
)

target_link_libraries(randomforest_test
    ${VIGRA_IMPEX_LIBRARY}
    ${HDF5_LIBRARIES}
)
