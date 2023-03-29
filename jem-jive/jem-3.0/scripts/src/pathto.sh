
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: $(basename "$0") [OPTIONS] [--] [DIR] FILE

Prints the relative path from the directory DIR to the file or
directory named FILE. If DIR is not specified, the relative path
from the current working directory will be printed.

Options:

  -h, --help : show this help.

EOF
}


#========================================================================
#   Main script
#========================================================================


curdir=""
target=""
fpath=""
tail=""

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

if [ $# -gt 2 ]; then
  echo "$0: too many arguments; try \`$0 --help'." 1>&2
  exit 1
fi

if [ $# -gt 1 ]; then
  cd "$1" || exit 1
  shift
fi

target="$1"

if [ -z "$target" ]; then
  echo "$0: missing file; try \`$0 --help'." 1>&2
  exit 1
fi

curdir="$(pwd)"

if [ -d "$target" ]; then

  cd "$target" || exit 1

  target="$(pwd)"

  cd "$curdir" || exit 1

else

  cd "$(dirname "$target")" || exit 1

  fpath="$(pwd)"
  target="${fpath%/}/$(basename "$target")"

  cd "$curdir" || exit 1

fi

tail="${target#$curdir}"

if [ "xx$tail" != "xx$target" ]; then

  if [ -z "$tail" ]; then
    echo "."
  else
    echo "./${tail#/}"
  fi

  exit 0

fi

fpath=""
curdir="${curdir#*/}"

while [ -n "$curdir" ]; do

  cd ".." || exit 1

  if [ -z "$fpath" ]; then
    fpath=".."
  else
    fpath="$fpath/.."
  fi

  curdir="$(pwd)"
  tail="${target#$curdir}"
  curdir="${curdir#*/}"

  if [ "xx$tail" != "xx$target" ]; then

    if   [ -z "$tail" ]; then
      echo "$fpath"
    elif [ -z "$curdir" ]; then
      echo "$target"
    else
      echo "$fpath/${tail#/}"
    fi

    exit 0

  fi

done

echo "$1"
