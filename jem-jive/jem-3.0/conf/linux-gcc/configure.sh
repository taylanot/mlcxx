
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   ConfigureLinuxGCC_ia32 ()
#------------------------------------------------------------------------


ConfigureLinuxGCC_ia32()
{
  local march=""
  local mtune=""
  local mmath=""

  if [ "xx$CPU_TYPE" = "xxcrusoe" ]; then

    march="i586"

  else

    march="$(uname -m)"

    case "$march" in

      i486|i586|i686)    ;;

      x86_64)            march="i686" ;;

      *)                 march="i486" ;;

    esac

  fi

  mtune=$march

  if [ -n "$CPU_HAS_SSE" ]; then

    mmath="-mfpmath=sse -msse"

    case $CPU_VENDOR in

      intel) mtune=pentium3 ;;
      amd)   mtune=athlon   ;;

    esac

  fi

  if [ -n "$CPU_HAS_SSE2" ] && [ $CXX_VERSION -ge 30100 ]; then

    case $CPU_VENDOR in

      intel) mtune=pentium4 ;;
      amd)   mtune=athlon-4 ;;

    esac

    case $CPU_TYPE in

      pentium-m)

        if [ $CXX_VERSION -gt 30400 ]; then
          mtune=pentium-m
        else
          mtune=pentium3
        fi ;;

    esac

  fi

  if [ -n "$CPU_HAS_SSE3" ] && [ $CXX_VERSION -gt 30400 ]; then

    case $CPU_VENDOR in

      intel) mtune=prescott ;;

    esac

    case $CPU_TYPE in

      pentium-m) mtune=pentium-m ;;

    esac

  fi

  if [ $CXX_VERSION -ge 40200 ]; then

    mtune=native

  fi

  if [ $CXX_VERSION -ge 30100 ]; then

    if [ $CXX_VERSION -gt 30400 ]; then
      mtune="-mtune=$mtune"
    else
      mtune="-mcpu=$mtune"
    fi

    CXX_STD_FLAGS="$CXX_STD_FLAGS -march=$march $mtune $mmath"

  fi
}


#------------------------------------------------------------------------
#   ConfigureLinuxGCC_x86_64
#------------------------------------------------------------------------


ConfigureLinuxGCC_x86_64()
{
  local mtune=""
  local mbits=""

  if [ -z "$USR_BIT_MODE" ]; then
    mbits="-m64"
  else
    mbits="-m$USR_BIT_MODE"
  fi

  printf "Testing if g++ supports the \`$mbits' option ... "

  if TestCompiler "$CXX" "$mbits" ; then

    echo "yes."

    CXX_STD_FLAGS="$CXX_STD_FLAGS $mbits"

  else

    echo "no."

    mbits=""

  fi

  case "$mbits" in

    -m64)

      SYS_BITS=64
      CXX_STD_FLAGS="$CXX_STD_FLAGS -msse2"

      LogEcho "Compiling in 64-bit mode" \
              "(specify \`-32' for 32-bit mode)." ;;

    -m32)

      SYS_BITS=32

      LogEcho "Compiling in 32-bit mode" \
              "(specify \`-64' for 64-bit mode)."

      ConfigureLinuxGCC_ia32 ;;

    *)

      LogEcho "Compiling in the default bit mode." ;;

  esac

  if [ "$SYS_BITS" = "64" ] && [ $CXX_VERSION -gt 30400 ]; then

    if [ $CXX_VERSION -ge 40200 ]; then

      mtune=native

    else

      case $CPU_VENDOR in

        intel)  mtune=nocona  ;;
        amd)    mtune=opteron ;;

      esac

    fi

    if [ -n $mtune ]; then
      CXX_STD_FLAGS="$CXX_STD_FLAGS -mtune=$mtune"
    fi

  fi
}


#------------------------------------------------------------------------
#   ConfigureLinuxGCC ()
#------------------------------------------------------------------------


ConfigureLinuxGCC()
{
  local oflags="-DNDEBUG -funroll-loops"

  CXX_DBG_FLAGS="-g"
  CXX_PRF_FLAGS="-pg"
  CXX_OPT_LEVEL="3"
  CXX_BLD_FLAGS="-Wconversion"
  CXX_OPT_FLAGS1="-O1 $oflags"
  CXX_OPT_FLAGS2="-O2 $oflags"
  CXX_OPT_FLAGS3="-O3 $oflags"

  if [ "xx$CXX_NAME2" != "xxclang" ]; then

    CXX_BLD_FLAGS="$CXX_BLD_FLAGS -finline-functions"

    if [ $CXX_VERSION -ge 50000 ]; then
      CXX_BLD_FLAGS="$CXX_BLD_FLAGS -Wsuggest-override"
    fi

  fi

  if [ "xx$SYS_NAME" = "xxcygwin" ]; then
    CXX_LNK_FLAGS=""
    CXX_DLL_FLAGS=""
  else
    CXX_LNK_FLAGS="-rdynamic"
    CXX_DLL_FLAGS="-fPIC"
  fi

  if [ $CXX_VERSION -ge 40000 ]; then

    LogEcho "Enabled vectorization support."

    CXX_OPT_FLAGS3="$CXX_OPT_FLAGS3 -ftree-vectorize"

  fi

  if [ -z "$USR_CXX_FLAGS" ]; then

    CXX_STD_FLAGS="-O -Wall -fstrict-aliasing"

    case "$SYS_ARCH" in

      ia32)    ConfigureLinuxGCC_ia32 ;;

      x86_64)  ConfigureLinuxGCC_x86_64 ;;

    esac

  fi

  if [ $CXX_VERSION -lt 60100 ]; then
    CXX_STD_FLAGS="$CXX_STD_FLAGS -std=c++11"
  fi

  if [ "xx$SYS_NAME" = "xxlinux" ] && [ "$SYS_BITS" = "32" ]
  then
    CXX_BLD_FLAGS="$CXX_BLD_FLAGS -D_FILE_OFFSET_BITS=64"
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

ConfigureLinuxGCC
