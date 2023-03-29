
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   IdentifyPosixCompiler ()
#------------------------------------------------------------------------


IdentifyPosixCompiler()
{
  local src="ccident.cpp"
  local exe="ccident${EXE_SUFFIX}"
  local cmd="$CXX -o $exe $src"
  local i=""

  cat<<EOF > "$src"

#include "include/jem/ccident.h"

extern "C"
{
  #include <stdio.h>
}

int main ()
{
#ifdef JEM_CXX_NAME
  ::printf ( "name=%s\n", JEM_CXX_NAME );
#endif
#ifdef JEM_CXX_NAME2
  ::printf ( "name2=%s\n", JEM_CXX_NAME2 );
#endif
#ifdef JEM_CXX_VERSION
  ::printf ( "version=%ld\n", (long) JEM_CXX_VERSION );
#endif
  return 0;
}

EOF

  eval echo '"Executing \`"'"${cmd}"\"\'' ..." >> "$JEM_LOGFILE"'

  eval "$cmd" '>> "$JEM_LOGFILE" 2>&1'

  if [ -x "$exe" ]; then

    for i in $(./$exe); do

      case "$i" in

        name=*)    CXX_NAME="${i#name=}" ;;
        name2=*)   CXX_NAME2="${i#name2=}" ;;
        version=*) CXX_VERSION="${i#version=}" ;;

      esac

    done

    rm -f -- "$exe"
    rm -f -- "ccident${OBJ_SUFFIX}"

  fi

  rm -f -- "$src"
}


#========================================================================
#   Main script
#========================================================================


IdentifyPosixCompiler
