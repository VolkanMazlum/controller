
# custom_stdp_module/CMakeLists.txt
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

# This CMakeLists.txt is configured to build your external module for NEST. For
# illustrative reasons this module is called 'my' (change SHORT_NAME to your
# preferred module name). NEST requires you to extend the 'SLIModule' (see
# mymodule.h and mymodule.cpp as an example) and provide a module header
# (see MODULE_HEADER). The subsequent instructions
#
# The configuration requires a compiled and installed NEST; if `nest-config` is
# not in the PATH, please specify the absolute path with `-Dwith-nest=...`.
#
# For more informations on how to extend and use your module see:
#           https://nest.github.io/nest-simulator/extension_modules

# 1) Name your module here, i.e. add later with -Dexternal-modules=${moduleName}:
set( SHORT_NAME custom_stdp_module )

#    the complete module name is here:
set( MODULE_NAME ${SHORT_NAME} )

# 2) Add all your sources here
set( MODULE_SOURCES
    custom_stdp_module.cpp

      eglif_io_nestml.cpp eglif_io_nestml.h
      eglif_pc_nestml.cpp eglif_pc_nestml.h
      
      eglif_mli.cpp eglif_mli.h
      eglif_dcnp.cpp eglif_dcnp.h
      stdp_synapse_sinexp.h stdp_synapse_sinexp.cpp
      
      stdp_synapse_alpha.h stdp_synapse_alpha.cpp
      extended_post_history_archiving_node.cpp extended_post_history_archiving_node.h
      extended_hist_entry.cpp extended_hist_entry.h

      eglif_cond_alpha_multisyn.cpp eglif_cond_alpha_multisyn.h

      tracking_neuron_nestml.cpp tracking_neuron_nestml.h
      custom_stdp_module.cpp custom_stdp_module.h
      basic_neuron_nestml.cpp basic_neuron_nestml.h
      diff_neuron_nestml.cpp diff_neuron_nestml.h
      rb_neuron_nestml.cpp rb_neuron_nestml.h
      tracking_neuron_planner_nestml.cpp tracking_neuron_planner_nestml.h
      state_neuron_nestml.cpp state_neuron_nestml.h

    )

# 3) We require a header name like this:
set( MODULE_HEADER ${MODULE_NAME}.h )
# containing the class description of the class extending the SLIModule

# 4) Specify your module version
set( MODULE_VERSION_MAJOR 1 )
set( MODULE_VERSION_MINOR 0 )
set( MODULE_VERSION "${MODULE_VERSION_MAJOR}.${MODULE_VERSION_MINOR}" )

# Leave the call to "project(...)" for after the compiler is determined.

# Set the `nest-config` executable to use during configuration.
set( with-nest OFF CACHE STRING "Specify the `nest-config` executable." )

# If it is not set, look for a `nest-config` in the PATH.
if ( NOT with-nest )
    # try find the program ourselves
    find_program( NEST_CONFIG
          NAMES nest-config
        )
    if ( NEST_CONFIG STREQUAL "NEST_CONFIG-NOTFOUND" )
        message( FATAL_ERROR "Cannot find the program `nest-config`. Specify via -Dwith-nest=... ." )
    endif ()
else ()
    set( NEST_CONFIG ${with-nest} )
endif ()

# Use `nest-config` to get the compile and installation options used with the
# NEST installation.

# Get the compiler that was used for NEST.
execute_process(
    COMMAND ${NEST_CONFIG} --compiler
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_COMPILER
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# One check on first execution, if `nest-config` is working.
if ( NOT RES_VAR EQUAL 0 )
    message( FATAL_ERROR "Cannot run `${NEST_CONFIG}`. Please specify correct `nest-config` via -Dwith-nest=... " )
endif ()

# Setting the compiler has to happen before the call to "project(...)" function.
set( CMAKE_CXX_COMPILER "${NEST_COMPILER}" )

project( ${MODULE_NAME} CXX )

# Get the install prefix.
execute_process(
    COMMAND ${NEST_CONFIG} --prefix
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the CXXFLAGS.
execute_process(
    COMMAND ${NEST_CONFIG} --cflags
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the Includes.
execute_process(
    COMMAND ${NEST_CONFIG} --includes
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if ( NEST_INCLUDES )
    # make a cmake list
    string( REPLACE " " ";" NEST_INCLUDES_LIST "${NEST_INCLUDES}" )
    foreach ( inc_complete ${NEST_INCLUDES_LIST} )
        # if it is actually a -Iincludedir
        if ( "${inc_complete}" MATCHES "^-I.*" )
            # get the directory
            string( REGEX REPLACE "^-I(.*)" "\\1" inc "${inc_complete}" )
            # and check whether it is a directory
            if ( IS_DIRECTORY "${inc}" )
                include_directories( "${inc}" )
            endif ()
        endif ()
    endforeach ()
endif ()

# Get, if NEST is build as a (mostly) static application. If yes, also only build
# static library.
execute_process(
    COMMAND ${NEST_CONFIG} --static-libraries
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_STATIC_LIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if ( NEST_STATIC_LIB )
    set( BUILD_SHARED_LIBS OFF )
else ()
    set( BUILD_SHARED_LIBS ON )
endif ()

# Get all linked libraries.
execute_process(
    COMMAND ${NEST_CONFIG} --libs
    RESULT_VARIABLE RES_VAR
    OUTPUT_VARIABLE NEST_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# on OS X
set( CMAKE_MACOSX_RPATH ON )


if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # Use the `NEST_PREFIX` as `CMAKE_INSTALL_PREFIX`.
    set( CMAKE_INSTALL_PREFIX ${NEST_PREFIX} CACHE STRING "Install path prefix, prepended onto install directories." FORCE )

    # Retrieve libs folder in nest
    execute_process(
        COMMAND ${NEST_CONFIG} --libdir
        RESULT_VARIABLE RES_VAR
        OUTPUT_VARIABLE NEST_LIBDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Append lib/nest to the install_dir
    set( CMAKE_INSTALL_LIBDIR "${NEST_LIBDIR}/nest" CACHE STRING "object code libraries (lib/nest or lib64/nest or lib/<multiarch-tuple>/nest on Debian)" FORCE )
else()
    # Check If CMAKE_INSTALL_PREFIX is not empty string
    if("${CMAKE_INSTALL_PREFIX}" STREQUAL "")
        message(FATAL_ERROR "CMAKE_INSTALL_PREFIX cannot be an empty string")
    endif()

    # Set lib folder to the given install_dir
    set( CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX} CACHE STRING "object code libraries (lib/nest or lib64/nest or lib/<multiarch-tuple>/nest on Debian)" FORCE )
endif()


include( GNUInstallDirs )

# CPack stuff. Required for target `dist`.
set( CPACK_GENERATOR TGZ )
set( CPACK_SOURCE_GENERATOR TGZ )

set( CPACK_PACKAGE_DESCRIPTION_SUMMARY "NEST Module ${MODULE_NAME}" )
set( CPACK_PACKAGE_VENDOR "NEST Initiative (http://www.nest-initiative.org/)" )

set( CPACK_PACKAGE_VERSION_MAJOR ${MODULE_VERSION_MAJOR} )
set( CPACK_PACKAGE_VERSION_MINOR ${MODULE_VERSION_MINOR} )
set( CPACK_PACKAGE_VERSION ${MODULE_VERSION} )

set( CPACK_SOURCE_IGNORE_FILES
    "\\\\.gitignore"
    "\\\\.git/"
    "\\\\.travis\\\\.yml"

    # if we have in source builds
    "/build/"
    "/_CPack_Packages/"
    "CMakeFiles/"
    "cmake_install\\\\.cmake"
    "Makefile.*"
    "CMakeCache\\\\.txt"
    "CPackConfig\\\\.cmake"
    "CPackSourceConfig\\\\.cmake"
    )
set( CPACK_SOURCE_PACKAGE_FILE_NAME ${MODULE_NAME} )

set( CPACK_PACKAGE_INSTALL_DIRECTORY "${MODULE_NAME} ${MODULE_VERSION}" )
include( CPack )

# add make dist target
add_custom_target( dist
    COMMAND ${CMAKE_MAKE_PROGRAM} package_source
    # not sure about this... seems, that it will be removed before dist...
    # DEPENDS doc
    COMMENT "Creating a source distribution from ${MODULE_NAME}..."
    )


if ( BUILD_SHARED_LIBS )
    # When building shared libraries, also create a module for loading at runtime
    # with the `Install` command.
    add_library( ${MODULE_NAME}_module MODULE ${MODULE_SOURCES} )
    set(LINKER_FLAGS "${NEST_LIBS}")
    if ( APPLE )
        set(LINKER_FLAGS "${LINKER_FLAGS} -undefined dynamic_lookup")
    endif ()
    set_target_properties( ${MODULE_NAME}_module
        PROPERTIES
        COMPILE_FLAGS "${NEST_CXXFLAGS} -DLTX_MODULE"
        LINK_FLAGS "${LINKER_FLAGS}"
        PREFIX ""
        OUTPUT_NAME ${MODULE_NAME} )
    install( TARGETS ${MODULE_NAME}_module
        DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ()

message( "" )
message( "-------------------------------------------------------" )
message( "${MODULE_NAME} Configuration Summary" )
message( "-------------------------------------------------------" )
message( "" )
message( "C++ compiler         : ${CMAKE_CXX_COMPILER}" )
message( "Build static libs    : ${NEST_STATIC_LIB}" )
message( "C++ compiler flags   : ${CMAKE_CXX_FLAGS}" )
message( "NEST compiler flags  : ${NEST_CXXFLAGS}" )
message( "NEST include dirs    : ${NEST_INCLUDES}" )
message( "NEST libraries flags : ${NEST_LIBS}" )
message( "" )
message( "-------------------------------------------------------" )
message( "" )
message( "You can now build and install '${MODULE_NAME}' using" )
message( "  make" )
message( "  make install" )
message( "" )
message( "The library file lib${MODULE_NAME}.so will be installed to" )
message( "  ${CMAKE_INSTALL_FULL_LIBDIR}" )
message( "The module can be loaded into NEST using" )
message( "  (${MODULE_NAME}) Install       (in SLI)" )
message( "  nest.Install(${MODULE_NAME})   (in PyNEST)" )
message( "" )
