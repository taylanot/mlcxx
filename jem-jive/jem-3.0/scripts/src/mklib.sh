
#@include "${JEMDIR:?}/scripts/include/config.sh"
#@include "${JEMDIR:?}/scripts/include/utils.sh"
#@include "${JEMDIR:?}/conf/${SYS_NAME:?}-${CXX_NAME:?}/MakeLib.sh"

#@skip

. "${JEMDIR:?}/scripts/include/config.sh"
. "${JEMDIR:?}/scripts/include/utils.sh"
. "${JEMDIR:?}/conf/${SYS_NAME:?}-${CXX_NAME:?}/MakeLib.sh"

#@endskip

#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: $(basename "$0") [OPTIONS] [--] LIB FILES

Updates the static library LIB with the object files FILES.

Options:

  -h, --help : show this help.

EOF
}


#------------------------------------------------------------------------
#   Run ( [-h|--help] [--] %lib [%file1 %file2 ...] )
#------------------------------------------------------------------------


Run()
{
  while [ $# -gt 0 ]; do

    case "$1" in

      -h|--help)

        PrintHelp
        exit 0 ;;

      --)

        shift
        break ;;

      -*)

        echo "$0: unknown option: $1; try \`$0 --help'." 1>&2
        exit 1 ;;

      *)

        break ;;

    esac

    shift

  done

  if [ $# -lt 1 ]; then
    echo "$0: missing library name." 1>&2
    exit 1
  fi

  if [ $# -lt 2 ]; then
    exit 0
  fi

  MakeLib "$@"
}


#========================================================================
#   Main script
#========================================================================


Run "$@"
