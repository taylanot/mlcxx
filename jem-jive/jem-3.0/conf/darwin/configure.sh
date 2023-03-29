
. "${JEMDIR:?}/conf/posix/configure.sh"

#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   ConfigureDarwin ()
#------------------------------------------------------------------------


ConfigureDarwin()
{
  if [ -z "$CXX" ]; then
    CXX="g++"
    CXX_NAME="gcc"
  fi

  AR="ar -rucs"
  SYS_ARCH="ia32"
  SYS_IS_UNIX98=1

  CPU_TYPE=""

  if FindExe --silent machine; then

    case $(machine) in

      ppc970)  CPU_TYPE="g5"
               SYS_ARCH="ppc" ;;

      i?86)    CPU_TYPE="x86"
               SYS_ARCH="ia32" ;;

    esac

  fi

  Log "CPU Type: $CPU_TYPE"
}


#========================================================================
#   Main script
#========================================================================

echo "Configuring for Darwin (MacOS X)."

ConfigureDarwin
