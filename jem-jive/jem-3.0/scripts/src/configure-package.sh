
. "${JEMDIR:?}/scripts/include/error.sh"
. "${JEMDIR:?}/scripts/include/utils.sh"
. "${JEMDIR:?}/scripts/include/conf-utils.sh"


#========================================================================
#   Global variables
#========================================================================


VERBOSE=""

PKG_NAME=""
PKG_DIR=""
PKG_INC_DIR=""

MY_LIBS=""
MY_LIBDIRS=""
MY_INCDIRS=""

EXTRA_MAKE_DEFS=""


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  local f=""

  if [ -z "$JEM_SUBCONF" ]; then

    cat<<EOF

Usage: $(basename "$0") [OPTIONS] [PKG]

Configures the package PKG.
The package name may be omitted if the path name of the current
working directory contains a directory named \`packages'.

Options:

  -h, --help            : show this help.
  -v, --verbose         : show what is going on.

EOF

  fi

  if [ -z "$PKG_NAME" ]; then
    echo "Specify a package name to get the full" \
         "list of options."
    echo ""
    return 0
  fi

  # Print system-specific help

  if f="$(FindSysconfFile print-help.sh)" ; then
    . "$f"
    echo ""
  fi
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

    esac

  done
}


#------------------------------------------------------------------------
#   PrintArgs ()
#------------------------------------------------------------------------


PrintArgs()
{
  local args="-h --help -v --verbose --incdirs --libdirs"
  local f=""

  # The system-specific arguments.

  if f="$(FindSysconfFile print-args.sh)" ; then
    args="$args $(. "$f")"
  fi

  echo "$args"
}


#------------------------------------------------------------------------
#   CheckDependencies ()
#------------------------------------------------------------------------


CheckDependencies()
{
  local p=""

  case "$PKG_NAME" in
    base|io|util) return ;;
  esac

  for p in $PKG_DEPS; do

    if [ -e "$JEMDIR/packages/$p/Makedefs.mk" ]; then
      continue;
    fi

    Log "Missing required package \`$p'."

    if [ -z "$JEM_SUBCONF" ]; then
      Error "The required package \`$p' has not yet been configured."
    else
      echo  "Missing the required package \`$p'." 1>&2
      echo  "Disabled the \`$PKG_NAME' package."  1>&2
    fi

    exit 1

  done

  return 0
}


#------------------------------------------------------------------------
#   PrintMakedefs ()
#------------------------------------------------------------------------


PrintMakedefs()
{
  local f="$(FindSysconfFile Makedefs.in)"

  if [ -z "$f" ]; then
    f="$JEMDIR/makefiles/pkg-defs.in"
  fi

  # Escape the '$' character.

  MY_LIBS="$(echo "$MY_LIBS" | sed 's/\$/\$\$/g')"
  MY_LIBDIRS="$(echo "$MY_LIBDIRS" | sed 's/\$/\$\$/g')"
  MY_INCDIRS="$(echo "$MY_INCDIRS" | sed 's/\$/\$\$/g')"

  Log "Makedefs source file: $f."

  SYS_NAME="$SYS_NAME" \
  SYS_ARCH="$SYS_ARCH" \
  PKG_NAME="$PKG_NAME" \
  PKG_DEPS="$PKG_DEPS" \
  PKG_DIR="$PKG_DIR" \
  MY_LIBS="$MY_LIBS" \
  MY_LIBDIRS="$MY_LIBDIRS" \
  MY_INCDIRS="$MY_INCDIRS" \
  EXTRA_MAKEDEFS="$EXTRA_MAKEDEFS" \
    "$JEMDIR/bin/shpp" "$f" || Die
}


#------------------------------------------------------------------------
#   ConfigurePackage ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


ConfigurePackage()
{
  local f="$(FindSysconfFile configure.sh)"

  if [ -n "$f" ]; then
    Log "Executing \`$f' ..."
    . "$f"
  fi

  WriteFile PrintMakedefs Makedefs.mk

  if [ -e DISABLED ]; then

    echo ""
    echo "The package \`$PKG_NAME' has been configured but not yet" \
         "enabled."
    echo "Run \`enable-package $PKG_NAME' to enable it."
    echo ""

    return 0

  fi

  if [ -e NoMakefile ]; then
    rm -f Makefile
    mv    NoMakefile Makefile
  fi

  if [ ! -d "$JEMDIR/makefiles/packages" ]; then
    MakeDir "$JEMDIR/makefiles/packages" || Die
  fi

  rm -f -- "$JEMDIR/makefiles/packages/${PKG_NAME}.mk"

  if [ -e Makedefs.mk ]; then
    cp -f -- Makedefs.mk \
             "$JEMDIR/makefiles/packages/${PKG_NAME}.mk" || Die
  fi
}


#------------------------------------------------------------------------
#   Run ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


Run()
{
  local name=""
  local args=""

  # Extract the package name

  while [ $# -gt 0 ]; do

    case "$1" in

      -*)

        args="$args $(quote "$1")" ;;

      *)

        if [ -z "$PKG_NAME" ]; then
          PKG_NAME="$1"
        else
          args="$args $(quote "$1")"
        fi ;;

    esac

    shift

  done

  eval set -- "$args"

  if [ -z "$PKG_NAME" ]; then

    PKG_NAME="$(GetPackageName)"

    if [ "xx$1" = "xx-h" ] || [ "xx$1" = "xx--help" ]; then
      PrintHelp
      exit 0
    fi

    if [ -z "$PKG_NAME" ]; then
      InputError "no package name specified."
    fi

  fi

  PKG_DIR="$JEMDIR/packages/$PKG_NAME"
  PKG_INC_DIR="$JEMDIR/include/jem/$PKG_NAME"

  if [ ! -d "$PKG_DIR" ]; then
    InputError "invalid package: $PKG_NAME."
  fi

  ChangeDir "$PKG_DIR" || Die

  if [ -e SPECS.sh ]; then

    name="$PKG_NAME"

    . ./SPECS.sh

    # Override whatever package name has been set in the SPECS file.

    PKG_NAME="$name"

  fi

  if [ -z "$JEM_SUBCONF" ]; then
    CheckArgs "$@"
  fi

  ParseArgs "$@"

  OpenLogfile "$(pwd)/configure.log"

  Log "Arguments: $@"

  if [ -z "$JEM_SUBCONF" ]; then
    UnconfigurePackage "$PKG_NAME"
  fi

  CheckDependencies
  ReadConfigData
  ConfigurePackage "$@"
}


#========================================================================
#   Main script
#========================================================================


Run "$@"
