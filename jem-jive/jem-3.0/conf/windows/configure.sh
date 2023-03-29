
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   ConfigureWindows ()
#------------------------------------------------------------------------


ConfigureWindows()
{
  local arch="$(uname -m)"

  AR="ar -rucs"

  OBJ_SUFFIX=".obj"
  EXE_SUFFIX=".exe"
  LIB_PREFIX=""
  LIB_SUFFIX=".lib"
  DLL_PREFIX=""
  DLL_SUFFIX=".dll"

  case "$arch" in

    i386|i486|i586|i686)

      SYS_ARCH="ia32"
      SYS_BITS=32 ;;

    x86_64)

      SYS_ARCH="x86_64"
      SYS_BITS="" ;;     # Not known yet; can be both 32 and 64.

  esac
}


#========================================================================
#   Main script
#========================================================================

echo "Configuring for Windows."

ConfigureWindows
