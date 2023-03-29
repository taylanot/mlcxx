
#@set TOPDIR="${TOPDIR:?}"
#@set TOOLKIT="${TOOLKIT:?}"

. "${JEMDIR:?}/scripts/include/error.sh"
. "${JEMDIR:?}/scripts/include/utils.sh"


#========================================================================
#   Global variables
#========================================================================


VERBOSE=""

PKG_NAME=""
PKG_DEPS=""
PKG_DESC=""


#========================================================================
#   Functions
#========================================================================


#@include $(eval "echo $TOPDIR/scripts/include/make-package.sh")

#@skip

. "${TOPDIR:?}/scripts/include/make-package.sh"

#@endskip


#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: $(basename "$0") [OPTIONS] [PKG]

Creates a new $TOOLKIT package named PKG.

Options:

  -h, --help    : show this help.
  -v, --verbose : show what is going on.

  --deps=LIST   : use LIST as the package dependency list.

  --desc=TEXT   : use TEXT as a brief description of the package.

EOF
}


#------------------------------------------------------------------------
#   ParseArgs ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


ParseArgs()
{
  local arg=""

  for arg in "$@"; do

    case "$arg" in

      -h|--help)

        PrintHelp
        exit 0 ;;

      -v|--verbose)

        VERBOSE=1 ;;

      --deps=*)

        PKG_DEPS="${arg#--deps=}" ;;

      --desc=*)

        PKG_DESC="${arg#--desc=}" ;;

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
    printf "Enter the package name: "
    read PKG_NAME
  fi

  if [ -z "$PKG_DEPS" ]; then
    printf "Enter the dependency list: "
    read PKG_DEPS
  fi

  if [ -z "$PKG_DESC" ]; then
    printf "Enter a brief description: "
    read PKG_DESC
  fi
}


#------------------------------------------------------------------------
#   PrintSpecsfile ()
#------------------------------------------------------------------------


PrintSpecsfile()
{
  cat<<EOF

PKG_NAME=$(quote "$PKG_NAME")
PKG_DEPS=$(quote "$PKG_DEPS")
PKG_DESC=$(quote "$PKG_DESC")

EOF
}


#------------------------------------------------------------------------
#   Run ()
#------------------------------------------------------------------------


Run()
{
  local pkgDir="$TOPDIR/packages/$PKG_NAME"

  if [ -e "$pkgDir" ]; then
    Error "The package \`$PKG_NAME' already exists".
  fi

  if [ -n "$VERBOSE" ]; then
    echo "Creating directory \`$PKG_DIR' ..."
  fi

  mkdir -- "$pkgDir"  || Die

  ChangeDir "$pkgDir" || Die

  if [ -n "$VERBOSE" ]; then
    echo "Writing \`NoMakefile' ..."
  fi

  PrintMakefile > NoMakefile

  if [ -n "$VERBOSE" ]; then
    echo "Writing \`SPECS.sh' ..."
  fi

  PrintSpecsfile > SPECS.sh

  if [ -n "$VERBOSE" ]; then
    echo "Creating subdirs ..."
  fi

  mkdir src   || Die
  mkdir doc   || Die
  mkdir conf  || Die

  if [ -n "$VERBOSE" ]; then
    echo "Creating include directory ..."
  fi

  mkdir -- "$TOPDIR/include/jem/$PKG_NAME" || Die

  if [ -n "$VERBOSE" ]; then
    echo "The package \`$PKG_NAME' is ready."
  fi
}


#========================================================================
#   Main script
#========================================================================


ParseArgs "$@"
Run       "$@"
