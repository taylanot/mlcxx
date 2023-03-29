
#@include "${JEMDIR:?}/scripts/include/config.sh"

#@skip

. "${JEMDIR:?}/scripts/include/config.sh"

#@endskip


#========================================================================
#   Global variables
#========================================================================


COPYRIGHT_FILE=""


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: $(basename "$0") [OPTIONS] [--] COPYRIGHT-FILE FILES

Adds the contents of COPYRIGHT-FILE to FILES.

Options:

  -h, --help    : show this help.
  -v, --verbose : show what is going on.

EOF
}


#------------------------------------------------------------------------
#   ProcessFile ( %file )
#------------------------------------------------------------------------


ProcessFile()
{
  local bak="${1}~"

  if [ -n "$VERBOSE" ]; then
    echo "Processing file \`$1' ..."
  fi

  if [ -f "$bak" ]; then
    rm -f -- "$bak"
  fi

  mv -- "$1" "$bak" || return 1

  if cat "$COPYRIGHT_FILE" "$bak" > "$1" ; then
    rm -f -- "$bak"
    return 0
  else
    mv -- "$bak" "$1"
    return 1
  fi
}


#------------------------------------------------------------------------
#   ProcessDir ( %dir )
#------------------------------------------------------------------------


ProcessDir()
{
  local cmd="find"
  local f=""

  if [ -n "$VERBOSE" ]; then
    echo "Processing directory \`$1' ..."
  fi

  if [ "$SYS_NAME" = "windows" ]; then
    f="$(command which grep)"
    f="$(dirname "$f")/find.exe"
    if [ -x "$f" ]; then
      cmd="$f"
    fi
  fi

  "$cmd" "$1" \( -name '*.h' -o -name '*.cpp' -o -name '*.tcc' -o \
                 -name '*.dox' \) -print | \
  while read f ; do
    if [ ! -d "$f" ]; then
      ProcessFile "$f"
    fi
  done
}


#------------------------------------------------------------------------
#   Run ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


Run()
{
  while [ $# -gt 0 ]; do

    case "$1" in

      -h|--help)

        PrintHelp
        exit 0 ;;

      -v|--verbose)

        VERBOSE=1 ;;

      --)

        shift
        break ;;

      *)

        break ;;

    esac

    shift

  done

  if [ $# -lt 1 ] || [ -z "$1" ]; then
    echo "$0: no copyright file specified; try \`$0 --help'." 1>&2
    exit 1
  fi

  COPYRIGHT_FILE="$1"
  shift

  if [ ! -e "$COPYRIGHT_FILE" ] || [ -d "$COPYRIGHT_FILE" ]; then
    echo "$0: invalid copyright file: $COPYRIGHT_FILE." 1>&2
    exit 1
  fi

  while [ $# -gt 0 ]; do

    if [ -z "$1" ]; then
      continue
    fi

    if   [ -d "$1" ]; then
      ProcessDir "$1"
    elif [ -e "$1" ]; then
      ProcessFile "$1"
    else
      echo "$0: no such file or directory: $1." 1>&2
      exit 1
    fi

    shift

  done
}


#========================================================================
#   Main script
#========================================================================

Run "$@"
