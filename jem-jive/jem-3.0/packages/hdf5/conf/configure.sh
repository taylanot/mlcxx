
#========================================================================
#   Global variables
#========================================================================


USE_HDF5=1

HDF5_LIBS="hdf5"
HDF5_LIBDIRS=""
HDF5_INCDIRS=""


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

      --hdf5-libs=?*)

        HDF5_LIBS="${arg#--hdf5-libs=}" ;;

      --hdf5-libdirs=?*)

        HDF5_LIBDIRS="${arg#--hdf5-libdirs=}" ;;

      --hdf5-incdirs=?*)

        HDF5_INCDIRS="${arg#--hdf5-incdirs=}" ;;

      --with-hdf5)

        USE_HDF5=1 ;;

      --without-hdf5)

        USE_HDF5="" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   TestHDF5 ()
#------------------------------------------------------------------------


TestHDF5()
{
  local f="$(FindSysconfFile test-hdf5.cpp)"

  if [ -z "$f" ]; then
    return 1
  fi

  MY_LIBS="$HDF5_LIBS"
  MY_LIBDIRS="$HDF5_LIBDIRS"
  MY_INCDIRS="$HDF5_INCDIRS"

  Log    "Testing HDF5 setup ..."

  printf "Testing if HDF5 works ... "

  if TestSyslibs "$f" ; then
    echo "yes."
    return 0
  else
    echo "no."
    return 1
  fi
}


#------------------------------------------------------------------------
# ConfigureSource()
#------------------------------------------------------------------------


ConfigureSource()
{
  local f=""
  local p=""

  PushDir "$PKG_DIR/src" || Die

  p="native/jemhdf5.h"

  if [ -e $p ]; then
    f="$p"
  fi

  p="native/jemhdf5-${SYS_NAME}.h"

  if [ -e $p ]; then
    f="$p"
  fi

  if [ -z "$f" ]; then
    Oops "Failed to find a suitable \`hdf5.h' header file."
  fi

  cp -f "$f" "jemhdf5.h" || Die

  PopDir || Die
}


#========================================================================
#   Main script
#========================================================================


SetDefaults "$@"
ParseArgs "$@"

if [ -n "$USE_HDF5" ] && TestHDF5 ; then

  if [ -n "$VERBOSE" ]; then

    echo ""
    echo "  --with-hdf5"
    echo "  --hdf5-libs=$(quote "$HDF5_LIBS")"
    echo "  --hdf5-libdirs=$(quote "$HDF5_LIBDIRS")"
    echo "  --hdf5-incdirs=$(quote "$HDF5_INCDIRS")"
    echo ""

  fi

  Log "Enabled HDF5 support."

  MY_LIBS="$HDF5_LIBS"
  MY_LIBDIRS="$HDF5_LIBDIRS"
  MY_INCDIRS="$HDF5_INCDIRS"

  ConfigureSource

else

  LogEcho "Disabled the hdf5 package."

  exit 0

fi
