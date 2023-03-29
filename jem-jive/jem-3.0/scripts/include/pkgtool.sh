
#@include "${JEMDIR:?}/scripts/include/error.sh"
#@include "${JEMDIR:?}/scripts/include/utils.sh"

#@skip

if [ -n "$JEM_SCRIPTS_PKGTOOL_SH" ]; then
  return 0
else
  JEM_SCRIPTS_PKGTOOL_SH=1
fi

. "${JEMDIR:?}/scripts/include/error.sh"
. "${JEMDIR:?}/scripts/include/utils.sh"

#@endskip


#========================================================================
#   Global variables
#========================================================================


VERBOSE=""
PKG_DIR=""
PKG_NAME=""
PKG_INC_DIR=""


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: $(basename "$0") [OPTIONS] [PKG]

$(PrintInfo PKG)
The package name may be omitted if the full name of the current
working directory contains a directory named \`packages'.

Options:

  -h, --help    : show this help.
  -v, --verbose : show what is going on.

EOF
}


#------------------------------------------------------------------------
#   ParseArgs ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


ParseArgs()
{
  local arg=""

  for arg in "$@" ; do

    case "$arg" in

      -h|--help)

        PrintHelp
        exit 0 ;;

      -v|--verbose)

        VERBOSE=1 ;;

      -*)

        OptionError "$arg" ;;

      *)

        if [ -n "$PKG_NAME" ]; then
          OptionError "$arg"
        fi

        PKG_NAME="$arg" ;;

    esac

  done

  if [ -z "$PKG_NAME" ]; then

    PKG_NAME="$(GetPackageName)"

    if [ -z "$PKG_NAME" ]; then
      echo "$0: failed to determine the current package name." 1>&2
      exit 1
    fi

  fi
}


#========================================================================
#   Main script
#========================================================================


ParseArgs "$@"

PKG_DIR="${TOPDIR:?}/packages/$PKG_NAME"
PKG_INC_DIR="${TOPDIR:?}/include/$(ToLower "$TOOLKIT")/$PKG_NAME"

if [ ! -d "$PKG_DIR" ]; then
  echo "$0: unknown package: $PKG_NAME." 1>&2
  exit 1
fi

Run "$@"
