
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: $(basename "$0") [OPTIONS] [--] [DIR]

Prints the name of the package containing the directory DIR. If no
directory is specified, the current working directory will be used.

Options:

  --help, -h : show this help.

EOF
}


#========================================================================
#   Main script
#========================================================================


dir=""

while [ $# -gt 0 ]; do

  case "$1" in

    -h|--help)

      PrintHelp
      exit 0 ;;

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

if [ -n "$1" ]; then
  dir="/${1#/}"
else
  dir="$(pwd)"
fi

case "$dir" in

  */packages/*)

    dir="${dir##*/packages/}"
    dir="${dir%%/*}" ;;

  *)

    dir="" ;;

esac

if [ -z "$dir" ]; then
  exit 1
fi

echo "$dir"
