
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

Prints a prefix string that can be prepended to the object files in
the directory DIR to make them unique within an archive (library).
If no directory is specified, the current directory name is used to
generate the prefix string.

Options:

  --help, -h : show this help.

  -p         : include the parent of the "src" directory in the
               prefix string. This option has no effect if there
               is no "src" component in the directory path.

EOF
}


#========================================================================
#   Main script
#========================================================================


prefix=""
pflag=""
dir=""

while [ $# -gt 0 ]; do

  case "$1" in

    -h|--help)

      PrintHelp
      exit 0 ;;

    -p)

      pflag=1 ;;

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

dir="${dir%/}/"

if [ "xx$dir" = "xx/" ]; then
  exit 0
fi

prefix="${dir##*/[sS][rR][cC]/}"

if [ -n "$pflag" ]; then

  dir="${dir%/[sS][rR][cC]/*}"

  if [ "xx$dir" != "xx$prefix" ]; then
    prefix="${dir##*/}/$prefix"
  fi

fi

echo "${prefix#/}" | sed -e 's/[./\[:space:]-]/_/g'
