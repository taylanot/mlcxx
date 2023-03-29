
. "${JEMDIR:?}/scripts/include/config.sh"
. "${JEMDIR:?}/scripts/include/error.sh"
. "${JEMDIR:?}/scripts/include/utils.sh"
. "${JEMDIR:?}/conf/${SYS_NAME:?}-${CXX_NAME:?}/MakeLib.sh"


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   Run ( [--libdir=%libdir] [--libpath=%libpath] [--] [%objdir] )
#------------------------------------------------------------------------


Run()
{
  local ispath=""
  local libdir=""
  local objdir=""
  local objects=""
  local lib=""
  local pkg=""
  local f=""

  while [ $# -gt 0 ]; do

    case "$1" in

      --libdir=*)

        libdir="${1#--libdir=}" ;;

      --libpath=*)

        ispath=1
        libdir="${1#--libpath=}" ;;

      --)

        shift
        break ;;

      -*)

        OptionError "$1" ;;

      *)

        break ;;

    esac

    shift

  done

  if [ -z "$libdir" ]; then
    Oops "No output directory specified."
  fi

  if [ $# -lt 1 ] || [ -z "$1" ]; then
    Oops "No object directory specified."
  fi

  objdir="$1"
  shift

  if [ $# -gt 0 ]; then
    Oops "Invalid option: $1."
  fi

  if [ ! -d "$objdir" ]; then
    Oops "No such directory: $objdir."
  fi

  if [ ! -d "$libdir" ]; then
    mkdir -- "$libdir" || Die
  fi

  # Determine the library path name.

  if ! pkg="$(GetPackageName)" ; then
    Oops "Failed to determine the current package name."
  fi

  case "$pkg" in

    base|io|util|std)  lib="jem"  ;;

    *)                 lib="jem$pkg" ;;

  esac

  lib="$libdir/${LIB_PREFIX}${lib}${LIB_SUFFIX}"

  if [ -n "$ispath" ]; then
    lib="../$lib"
  fi

  cd "$objdir" || Die

  for f in *"$OBJ_SUFFIX" ; do

    if [ ! -f "$f" ]; then
      continue
    fi

    objects="$objects $(quote "$f")"

  done

  if [ -z "$objects" ]; then
    exit 0
  fi

  eval "set -- $objects"

  echo "Adding object files in \`$objdir' to library:"

  MakeLib "$lib" "$@"
}


#========================================================================
#   Main
#========================================================================


Run "$@"
