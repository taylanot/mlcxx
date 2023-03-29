
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: $(basename "$0") [OPTIONS] [--] [DIRS]

Runs \`make' in the directories DIRS. If the directory list is
empty, \`make' is run in all subdirectories in the current
directory. Note that \`make' is only run in those directories
containing a file named \`Makefile' or \`makefile', and not
containing a file named \`DISABLED'.

Options:

  -h, --help    : show this help.

  -m CMD,
  --make=CMD    : use CMD to run \`make'. If not specified,
                  the contents of the \`MAKE' variable will be
                  used.

  -t NAME,
  --target=NAME : add NAME to the list of targets to be
                  updated by \`make'.

EOF
}


#------------------------------------------------------------------------
#   Run ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


Run()
{
  local targets=""
  local make="$MAKE"
  local dir=""

  while [ $# -gt 0 ] ; do

    case "$1" in

      -h|--help)

        PrintHelp
        exit 0 ;;

      -m)

        shift
        make="$1" ;;

      -m?*)

        make="${1#-m}" ;;

      --make=*)

        make="${1#--make=}" ;;

      -t)

        shift
        targets="$targets $1" ;;

      -t?*)

        targets="$targets ${1#-t}" ;;

      --target=*)

        targets="$targets ${1#--target=}" ;;

      --)

        shift
        break ;;

      -*)

        echo "$0: invalid option: $1; try \`$0 --help'." 1>&2
        exit 1 ;;

      *)

        break ;;

    esac

    shift

  done

  if [ -z "$make" ]; then
    echo "$0: no \`make' command specified." 1>&2
    exit 1
  fi

  if [ $# -lt 1 ]; then
    set -- *
  else
    set -- "$@"
  fi

  for dir in "$@"; do

    if [ ! -d "$dir" ]; then
      continue
    fi

    if [ -e "$dir/DISABLED" ]; then
      continue
    fi

    if [ -e "$dir/Makefile" ] || [ -e "$dir/makefile" ]; then
      eval "$make -C" \"\$dir\" "$targets" || exit 1
    fi
  done
}


#========================================================================
#   Main script
#========================================================================


Run "$@"
