
#@set TOPDIR="${TOPDIR:?}"
#@set TOOLKIT="${TOOLKIT:?}"

. "${JEMDIR:?}/scripts/include/error.sh"


#========================================================================
#   Global variables
#========================================================================


VERBOSE=""


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: $(basename "$0") [OPTIONS]

Installs the $TOOLKIT toolkit.

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

      *)

        OptionError "$arg" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   Run ()
#------------------------------------------------------------------------


Run()
{
  local pkg=""

  cd "$TOPDIR/packages" || exit 1

  for pkg in *; do

    if [ -d "$pkg" ] && [ ! -e "$pkg/DISABLED" ]; then

      if [ -n "$VERBOSE" ]; then
        echo "Installing package $pkg ..."
      fi

      "$TOPDIR/scripts/bin/install-package" "$pkg"

    fi

  done

  if [ -n "$VERBOSE" ]; then
    echo "$TOOLKIT has been installed."
  fi
}


#========================================================================
#   Main script
#========================================================================

ParseArgs "$@"
Run       "$@"
