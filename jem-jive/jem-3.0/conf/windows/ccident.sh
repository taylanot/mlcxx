
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   IdentifyWindowsCompiler ()
#------------------------------------------------------------------------


IdentifyWindowsCompiler()
{
  local src="ccident.cpp"
  local exe="ccident${EXE_SUFFIX}"
  local cmd=""
  local i=""

  case "$(ToLower "$CXX")" in

    *cl|*cl.exe)

      cmd="$CXX -Fe$exe $src" ;;

    *)

      cmd="$CXX -o  $exe $src" ;;

  esac

  cat<<EOF > "$src"

#include <cstdio>
#include "include/jem/ccident.h"

int main ()
{
#ifdef JEM_CXX_NAME
  std::printf ( "name=%s\n", JEM_CXX_NAME );
#endif
#ifdef JEM_CXX_NAME2
  std::printf ( "name2=%s\n", JEM_CXX_NAME2 );
#endif
#ifdef JEM_CXX_VERSION
  std::printf ( "version=%ld\n", (long) JEM_CXX_VERSION );
#endif
  return 0;
}

EOF

  eval echo '"Executing \`"'"${cmd}"\"\'' ..." >> "$JEM_LOGFILE"'

  eval "$cmd" '>> "$JEM_LOGFILE" 2>&1'

  if [ -x "$exe" ]; then

    for i in $(./$exe | tr -d '\r'); do

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


IdentifyWindowsCompiler
