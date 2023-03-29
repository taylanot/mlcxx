
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   ConfigureLinuxICC ()
#------------------------------------------------------------------------


ConfigureLinuxICC()
{
  local oflags="-DNDEBUG -unroll -ip"
  local i=""

  if [ -z "$USR_CXX_FLAGS" ]; then
    CXX_STD_FLAGS="-O -ansi_alias -restrict"
  fi

  CXX_LNK_FLAGS="-Xlinker --export-dynamic"
  CXX_DLL_FLAGS="-fPIC"
  CXX_DBG_FLAGS="-g"
  CXX_PRF_FLAGS="-p"
  CXX_OPT_LEVEL="2"
  CXX_BLD_FLAGS="-ip"

  case "$SYS_ARCH" in

    ia32|x86_64)

      if [ -n "$CPU_HAS_SSE2" ]; then

        LogEcho "Enabled vectorization support."

        oflags="$oflags -xW"

      fi ;;

    ia64)

      oflags="$oflags -ivdep_parallel -ftz" ;;

  esac

  CXX_OPT_FLAGS1="-O1 $oflags"
  CXX_OPT_FLAGS2="-O2 $oflags"
  CXX_OPT_FLAGS3="-O3 $oflags"
}


#========================================================================
#   Main script
#========================================================================

echo "Configuring for the Intel compiler (icpc)."

ConfigureLinuxICC
