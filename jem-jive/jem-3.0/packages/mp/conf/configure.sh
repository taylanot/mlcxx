
#========================================================================
#   Global variables
#========================================================================


USE_MPI=1

MPI_LIBS="mpi"
MPI_LIBDIRS=""
MPI_INCDIRS=""


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   SetDefaults ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


SetDefaults()
{
  local f="$(FindSysconfFile defaults.sh)"

  if [ -n "$f" ]; then
    Log "Executing \`$f' ..."
    . "$f"
  fi
}


#------------------------------------------------------------------------
#   ParseArgs ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


ParseArgs()
{
  local arg=""

  for arg in "$@" ; do

    case "$arg" in

      --mpi-libs=?*)

        MPI_LIBS="${arg#--mpi-libs=}" ;;

      --mpi-libdirs=?*)

        MPI_LIBDIRS="${arg#--mpi-libdirs=}" ;;

      --mpi-incdirs=?*)

        MPI_INCDIRS="${arg#--mpi-incdirs=}" ;;

      --with-mpi)

        USE_MPI=1 ;;

      --without-mpi)

        USE_MPI="" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   TestMPI ()
#------------------------------------------------------------------------


TestMPI()
{
  local f="$(FindSysconfFile test-mpi.cpp)"

  if [ -z "$f" ]; then
    return 1
  fi

  MY_LIBS="$MPI_LIBS"
  MY_LIBDIRS="$MPI_LIBDIRS"
  MY_INCDIRS="$MPI_INCDIRS"

  Log    "Testing MPI setup ..."

  printf "Testing if MPI works ... "

  if TestSyslibs "$f" ; then
    echo "yes."
    return 0
  else
    echo "no."
    return 1
  fi
}


#------------------------------------------------------------------------
#   ConfigureSource ()
#------------------------------------------------------------------------


ConfigureSource()
{
  local output="$PKG_INC_DIR/config.h"

  echo "Writing \`$output' ..."

  cat<<EOF > "$output"
// Generated on $(date).
// CHANGES TO THIS FILE WILL BE LOST.

#ifndef JEM_MP_CONFIG_H
#define JEM_MP_CONFIG_H

$(test -n "$USE_MPI" && echo "#define JEM_USE_MPI")

#endif
EOF

  PushDir "$PKG_DIR/src" || Die

  if [ -n "$USE_MPI" ]; then

    local f=""
    local p=""

    p="native/mpi.h"

    if [ -e $p ]; then
      f="$p"
    fi

    p="native/mpi-${SYS_NAME}.h"

    if [ -e $p ]; then
      f="$p"
    fi

    if [ -z "$f" ]; then
      Oops "Failed to find a suitable \`mpi.h' header file."
    fi

    cp -f "$f" "$PKG_INC_DIR/mpi.h" || Die

    if [ -e mpi/NoMakefile ]; then
      rm -f mpi/Makefile
      mv    mpi/NoMakefile mpi/Makefile || Die
    fi

  else

    if [ -e mpi/Makefile ]; then
      rm -f mpi/NoMakefile
      mv    mpi/Makefile mpi/NoMakefile || Die
    fi

  fi

  PopDir || Die
}


#========================================================================
#   Main script
#========================================================================


SetDefaults "$@"
ParseArgs "$@"

if [ -n "$USE_MPI" ] && TestMPI ; then

  if [ -n "$VERBOSE" ]; then

    echo ""
    echo "  --with-mpi"
    echo "  --mpi-libs=$(quote "$MPI_LIBS")"
    echo "  --mpi-libdirs=$(quote "$MPI_LIBDIRS")"
    echo "  --mpi-incdirs=$(quote "$MPI_INCDIRS")"
    echo ""

  fi

  Log "Enabled MPI support."

else

  USE_MPI=""
  MPI_LIBS=""
  MPI_LIBDIRS=""
  MPI_INCDIRS=""

  LogEcho "Disabled MPI support."

fi

ConfigureSource

MY_LIBS="$MPI_LIBS"
MY_LIBDIRS="$MPI_LIBDIRS"
MY_INCDIRS="$MPI_INCDIRS"
