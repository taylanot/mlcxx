
#========================================================================
#   Global variables
#========================================================================


VERBOSE=""
USR_JIVEDIR="$JIVEDIR"

export JIVEDIR="$(pwd)"
export JIVE_SUBCONF=1


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: configure [OPTIONS]

Configures the Jive library.

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

  for arg in "$@"; do

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
#   CheckJIVEDIR ()
#------------------------------------------------------------------------


CheckJIVEDIR()
{
  local a=""

  set -- $JIVEDIR

  if [ "xx$JIVEDIR" != "xx$1" ]; then

    cat<<EOF

The path name of the Jive installation directory seems to contains
spaces and/or special characters. This may result in problems when
compiling Jive-based programs. You are adviced to install Jem in a
directory with a path name that does not contain spaces and/or other
special characters. Alternatively, you can create a link to the Jive
directory and set the JIVEDIR environment variable to the name of that
link.

EOF

    printf "Press [RETURN] to continue."
    read a

  fi
}


#------------------------------------------------------------------------
#   PrintConfigData ()
#------------------------------------------------------------------------


PrintConfigData()
{
  echo "# Generated on $(date)."
  echo "# CHANGES TO THIS FILE WILL BE LOST."
  echo ""
}


#------------------------------------------------------------------------
#   InstallScripts ()
#------------------------------------------------------------------------


InstallScripts()
{
  local src=""
  local dest=""

  LogEcho "Installing scripts in \`scripts/bin' ..."

  export TOPDIR='"${JIVEDIR:?}"'
  export TOOLKIT="Jive"

  if [ -d scripts/bin ]; then
    rm -f -- scripts/bin/*
  else
    mkdir scripts/bin || Die
  fi

  PushDir "scripts/src" || Die

  for src in *.sh; do

    if [ ! -e "$src" ]; then
      continue
    fi

    dest="../bin/${src%.sh}"

    "$JEMDIR"/bin/shpp -x "$src" "$dest" || Die

  done

  PopDir || Die
}


#------------------------------------------------------------------------
#   ListPackages ()
#------------------------------------------------------------------------


ListPackages()
{
  local list=""
  local deps=""
  local p=""
  local i=""

  for p in packages/*; do

    if [ ! -d "$p" ] || [ -e "$p/DISABLED" ]; then
      continue
    fi

    p="$(basename "$p")"

    deps="$(ListPackageDeps $p)"

    for i in $deps; do
      if  ! FindWord --silent "$i" $list && \
        [ ! -e packages/$i/DISABLED ]
      then
        list="$list $i"
      fi
    done

  done

  echo "$list"
}


#------------------------------------------------------------------------
#   ConfigurePackages ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


ConfigurePackages()
{
  local list=""
  local p=""

  LogEcho "Configuring installed packages ..."
  echo    ""

  list=""

  for p in $(ListPackages); do

    echo "Configuring the package $p ..."

    Log  ""
    Log  "--- package $p ---"
    Log  ""

    if ! scripts/bin/configure-package "$@" $p; then
      list="$list $p"
    fi

  done

  LogEcho ""
  LogEcho "---"

  if [ -n "$list" ]; then

    echo ""
    echo "*** Configuration of the following packages failed:"
    echo ""
    echo "   " $list
    echo ""

  fi
}


#------------------------------------------------------------------------
#   Run ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


Run()
{
  ParseArgs "$@"

  CheckJIVEDIR
  CloseLogfile
  OpenLogfile "$(pwd)/configure.log"

  Log "Arguments: $@"

  UnconfigureJive

  WriteFile PrintConfigData scripts/include/config.sh

  InstallScripts
  ConfigurePackages "$@"

  CloseLogfile
}


#========================================================================
#   Main script
#========================================================================


. "$JIVEDIR"/scripts/include/conf-utils.sh
. "${JEMDIR:?}"/scripts/include/error.sh
. "${JEMDIR:?}"/scripts/include/utils.sh

echo ""

Run "$@" || Die

echo ""
echo "Finished the configuration of Jive."

if [ -n "$USR_JEMDIR" ] && [ -d "$USR_JEMDIR" ]; then

  PushDir "$USR_JEMDIR" || Die
  USR_JEMDIR="$(pwd)"
  PopDir                || Die

fi

if [ "xx$JIVEDIR" != "xx$USR_JIVEDIR" ]; then

  cat<<EOF

Programs based on Jive may use the environment variable JIVEDIR
to locate the Jive installation directory. It is a good idea to
set this variable in your shell start-up file. Insert the
following line if you are using a C-like shell (csh, tcsh):

  setenv JIVEDIR $(quote "$JIVEDIR")

Insert the following if you are using a Bourne-like shell
(ash, bash, ksh, zsh):

  export JIVEDIR=$(quote "$JIVEDIR")
EOF

fi

cat<<EOF

Now run \`make' to build the Jive libraries. Optionally, run
\`make debug' to build an additional set of libraries with
full debugging support. Note that these libraries require
quite a large amount of disk space.

Finally, run \`make clean' to delete temporary files.

EOF

