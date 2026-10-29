include(CheckCXXCompilerFlag)

set(KNOWHERE_WITH_PAG OFF)

if(__X86_64)
  set(_PAG_AVX512_OK ON)
  foreach(_pag_flag -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -mavx512vnni)
    string(REPLACE "-" "_" _pag_flag_var "${_pag_flag}")
    check_cxx_compiler_flag(${_pag_flag} PAG_FLAG${_pag_flag_var})
    if(NOT PAG_FLAG${_pag_flag_var})
      set(_PAG_AVX512_OK OFF)
    endif()
  endforeach()

  if(_PAG_AVX512_OK)
    set(KNOWHERE_WITH_PAG ON)
    add_library(pag STATIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pag/pag.cpp)
    set_target_properties(pag PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(pag PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pag)
    target_compile_features(pag PRIVATE cxx_std_17)
    target_compile_options(
      pag
      PRIVATE -mavx512f
              -mavx512dq
              -mavx512cd
              -mavx512bw
              -mavx512vl
              -mavx512vnni
              -Wno-unused-function
              -Wno-unused-variable
              -Wno-sign-compare
              -Wno-reorder
              -Wno-maybe-uninitialized)
    target_link_libraries(pag PUBLIC OpenMP::OpenMP_CXX)
    # GCC 9 keeps std::filesystem in libstdc++fs. Both the static library and
    # PagIndexNode use it, and the public link propagates to libknowhere.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
      target_link_libraries(pag PUBLIC stdc++fs)
    endif()
  endif()
endif()

message(STATUS "KNOWHERE_WITH_PAG=${KNOWHERE_WITH_PAG}")
