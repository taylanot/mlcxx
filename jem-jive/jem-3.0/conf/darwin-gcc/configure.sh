
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   ConfigureDarwinGCC ()
#------------------------------------------------------------------------


ConfigureDarwinGCC()
{
  local oflags="-DNDEBUG -funroll-loops"
  local mflags=""

  CXX_LNK_FLAGS="-Wl,-bind_at_load"
  CXX_DLL_FLAGS="-fPIC"
  CXX_DBG_FLAGS="-g"
  CXX_PRF_FLAGS="-pg"
  CXX_OPT_LEVEL="3"
  CXX_BLD_FLAGS="-Wconversion -Wno-deprecated"
  CXX_OPT_FLAGS1="-O1 $oflags"
  CXX_OPT_FLAGS2="-O2 $oflags"
  CXX_OPT_FLAGS3="-O3 $oflags"

  if [ "xx$CXX_NAME2" != "xxclang" ]; then
    CXX_BLD_FLAGS="$CXX_BLD_FLAGS -finline-functions"
  fi

  if [ -z "$USR_CXX_FLAGS" ]; then

    CXX_STD_FLAGS="-O -Wall -fstrict-aliasing"

    if [ $CXX_VERSION -lt 40000 ]; then
      CXX_STD_FLAGS="$CXX_STD_FLAGS -fcoalesce-templates"
    fi

    case "$CPU_TYPE" in

      g5)  mflags="-mcpu=G5 -mtune=G5" ;;

    esac

    if [ -n "$mflags" ] && TestCompiler "$CXX" $mflags; then
      CXX_STD_FLAGS="$CXX_STD_FLAGS $mflags"
    fi

  fi

  if [ $CXX_VERSION -lt 60100 ]; then
    CXX_STD_FLAGS="$CXX_STD_FLAGS -std=c++11"
  fi

  if [ $CXX_VERSION -ge 40000 ]; then

    LogEcho "Enabled vectorization support."

    CXX_OPT_FLAGS3="$CXX_OPT_FLAGS3 -ftree-vectorize"

  fi
}


#========================================================================
#   Main script
#========================================================================

if [ "xx$CXX_NAME2" = "xxclang" ]; then
  echo "Configuring for the GNU compiler (g++ [clang])."
else
  echo "Configuring for the GNU compiler (g++)."
fi

ConfigureDarwinGCC
