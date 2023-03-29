
. "${JEMDIR:?}/conf/posix/configure.sh"

#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   ReadCPUInfo_x86
#------------------------------------------------------------------------


ReadCPUInfo_x86()
{
  local procs=0
  local line=""
  local i=""

  CPU_TYPE=""
  CPU_VENDOR=""
  CPU_HAS_SSE=""
  CPU_HAS_SSE2=""
  CPU_HAS_SSE3=""

  Log "Reading CPU info ..."

  while read line; do

    line="$(ToLower "$line")"

    case "$line" in

      processor*)

        procs=$(( $procs + 1 )) ;;

      vendor_id*)

        Log "  $line"

        case "$line" in

          *genuinetm*)          CPU_VENDOR="transmeta" ;;
          *amd*)                CPU_VENDOR="amd"       ;;
          *intel*)              CPU_VENDOR="intel"     ;;

        esac ;;

      model?name*)

        Log "  $line"

        case "$line" in

          *crusoe*)             CPU_TYPE="crusoe"    ;;
          *pentium\(r\)\ m\ *)  CPU_TYPE="pentium-m" ;;
          *pentium*)            CPU_TYPE="pentium"   ;;
          *xeon*)               CPU_TYPE="xeon"      ;;
          *athlon*)             CPU_TYPE="athlon"    ;;
          *opteron*)            CPU_TYPE="opteron"   ;;
          *core\(tm\)2*)        CPU_TYPE="core2"     ;;

        esac ;;

      flags*)

        Log "  $line"

        for i in $line; do

          case "$i" in

            sse)  CPU_HAS_SSE=1  ;;
            sse2) CPU_HAS_SSE2=1 ;;
            sse3) CPU_HAS_SSE3=1 ;;

          esac

        done ;;

    esac

    # Quit after the first processor.

    if [ $procs -gt 1 ]; then
      break
    fi

  done

  if [ "zz$SYS_ARCH" = "zzx86_64" ]; then
    CPU_HAS_SSE3=1
  fi

  if [ -n "$CPU_HAS_SSE3" ]; then
    CPU_HAS_SSE2=1
  fi

  if [ -n "$CPU_HAS_SSE2" ]; then
    CPU_HAS_SSE=1
  fi

  Log "CPU type: $CPU_TYPE"
}


#------------------------------------------------------------------------
#   ConfigureLinux ()
#------------------------------------------------------------------------


ConfigureLinux()
{
  local arch="$(uname -m)"

  SYS_IS_UNIX98=1

  case "$arch" in

    i386|i486|i586|i686)

      SYS_ARCH="ia32"
      SYS_BITS=32 ;;

    x86_64)

      SYS_ARCH="x86_64"
      SYS_BITS="" ;;     # Not known yet; can be both 32 and 64.

    ia64)

      SYS_ARCH="ia64"
      SYS_BITS="64" ;;

  esac

  if [ -z "$CXX" ]; then

    if [ "xx$SYS_ARCH" = "xxia64" ]; then
      if FindExe --silent icpc; then
        CXX="icpc"
        CXX_NAME="icc"
      fi
    fi

    if [ -z "$CXX" ]; then
      CXX="g++"
      CXX_NAME="gcc"
    fi

  fi

  AR="ar -rucs"

  if [ -r /proc/cpuinfo ]; then

    case "$SYS_ARCH" in

      ia32|x86_64)  ReadCPUInfo_x86 < /proc/cpuinfo ;;

    esac

  fi
}


#========================================================================
#   Main script
#========================================================================

# This script configures both for Linux and Cygwin, as these two
# platforms are very similar.

if [ "xx$SYS_NAME" = "xxcygwin" ]; then
  echo "Configuring for Cygwin."
else
  echo "Configuring for Linux."
fi

ConfigureLinux
