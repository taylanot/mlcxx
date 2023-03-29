
. "${JEMDIR:?}/scripts/include/error.sh"
. "${JEMDIR:?}/scripts/include/utils.sh"
. "${JIVEDIR:?}/scripts/include/conf-utils.sh"


#========================================================================
#   Global variables
#========================================================================


VERBOSE=""

PKG_NAME=""
PKG_DIR=""
PKG_INC_DIR=""


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  if [ -z "$JIVE_SUBCONF" ]; then

    cat<<EOF

Usage: $(basename "$0") [OPTIONS] [PKG]

Configures the package PKG.
The package name may be omitted if the path name of the current
working directory contains a directory named \`packages'.

Options:

  -h, --help    : show this help.
  -v, --verbose : show what is going on.

EOF

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

      -*)

        OptionError "$arg" ;;

      *)

        if [ -n "$PKG_NAME" ]; then
          OptionError "$arg"
        fi

        PKG_NAME="$arg" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   CheckDependencies ()
#------------------------------------------------------------------------


CheckDependencies()
{
  local dir=""
  local p=""

  for p in $PKG_DEPS; do

    case "$p" in

      jem.*)

        dir="$JEMDIR/packages/${p#jem.}" ;;

      jive.*)

        dir="$JIVEDIR/packages/${p#jive.}" ;;

      *)

        dir="$JIVEDIR/packages/$p" ;;

    esac

    if [ -e "$dir/Makedefs.mk" ]; then
      continue
    fi

    Log "Missing required package \`$p'."

    if [ -z "$JIVE_SUBCONF" ]; then
      Error "The required package \`$p' has not yet been configured."
    else
      echo  "Missing the required package \`$p'." 1>&2
      echo  "Disabled the \`$PKG_NAME' package."  1>&2
    fi

    exit 1

  done
}


#------------------------------------------------------------------------
#   PrintMakedefs ()
#------------------------------------------------------------------------


PrintMakedefs()
{
  local PKG="$(ToUpper $PKG_NAME)"
  local f=""
  local p=""

  cat<<EOF

# Generated on $(date).
# CHANGES TO THIS FILE WILL BE LOST.

ifndef JIVE_${PKG}_MAKEDEFS_MK
       JIVE_${PKG}_MAKEDEFS_MK = 1

ifndef JIVEPATH
  JIVEPATH := \$(JIVEDIR)
endif

include \$(JIVEPATH)/makefiles/jive.mk

EOF

  for p in $PKG_DEPS; do

    case "$p" in

      jem.*)

        f="\$(JEMPATH)/makefiles/packages/${p#jem.}.mk" ;;

      jive.*)

        f="\$(JIVEPATH)/makefiles/packages/${p#jive.}.mk" ;;

      *)

        f="\$(JIVEPATH)/makefiles/packages/${p}.mk" ;;

    esac

    echo "include $f"

  done

  echo ""
  echo "JIVE_PACKAGES += $PKG_NAME"
  echo "JIVE_LIBS     := jive${PKG_NAME} \$(JIVE_LIBS)"
  echo ""
  echo "endif"
}


#------------------------------------------------------------------------
#   ConfigurePackage ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


ConfigurePackage()
{
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

  if [ ! -d "$JIVEDIR/makefiles/packages" ]; then
    MakeDir "$JIVEDIR/makefiles/packages" || Die
  fi

  rm -f -- "$JIVEDIR/makefiles/packages/${PKG_NAME}.mk"

  if [ -e Makedefs.mk ]; then
    cp -f "$JIVEDIR/packages/$PKG_NAME/Makedefs.mk" \
          "$JIVEDIR/makefiles/packages/${PKG_NAME}.mk" || Die
  fi
}


#------------------------------------------------------------------------
#   Run ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


Run()
{
  local name=""

  ParseArgs "$@"

  if [ -z "$PKG_NAME" ]; then
    if ! PKG_NAME="$(GetPackageName)"; then
      InputError "no package name specified."
    fi
  fi

  PKG_DIR="$JIVEDIR/packages/$PKG_NAME"
  PKG_INC_DIR="$JIVEDIR/include/jive/$PKG_NAME"

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

  OpenLogfile "$(pwd)/configure.log"

  Log "Arguments: $@"

  if [ -z "$JIVE_SUBCONF" ]; then
    UnconfigurePackage "$PKG_NAME"
  fi

  CheckDependencies
  ConfigurePackage "$@"
}


#========================================================================
#   Main script
#========================================================================


Run "$@"
