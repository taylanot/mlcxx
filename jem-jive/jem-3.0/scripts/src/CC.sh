
#@include "${JEMDIR:?}/scripts/include/config.sh"
#@include "${JEMDIR:?}/conf/${SYS_NAME:?}-${CXX_NAME:?}/CC.sh"

#@skip

. "${JEMDIR:?}/scripts/include/config.sh"
. "${JEMDIR:?}/conf/${SYS_NAME:?}-${CXX_NAME:?}/CC.sh"

#@endskip


#========================================================================
#   Global variables
#========================================================================


LIBS=""
LIBDIRS=""
INCDIRS=""
INPUT=""
OUTPUT=""
FLAGS=""
NO_EXEC=""
COMPILE_ONLY=""


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: $(basename "$0") [OPTIONS] [--] FILES

Compiles and or links FILES.

Options:

  -h, --help     : show this help.
  -v, --version  : print the compiler version.
  -V, --info     : print information about the native compiler and
                 : the standard flags.

  -n             : show the command that would be executed, but
                   do nothing.

  -c             : compile but do not link.

  -shared        : create a shared object (DLL).

  -o FILE        : write the compiler output to FILE.

  -f FLAGS       : pass FLAGS directly to the native compiler.

  -f-            : do not use the standard compiler flags.

  -g             : add debugging information to the compiled
                   object files.

  -p             : add profiling instructions to the compiled
                   object files.

  -O0            : disable optimizations.
  -O1            : perform simple optimizations.
  -O2            : perform more elaborate optimizations.
  -O3            : perform aggressive optimizations.
  -O             : perform optimizations at the default level.

  -I DIRS        : search in DIRS for header filers.
  -L DIRS        : search in DIRS for libraries.
  -l LIBS        : link with the libraries in LIBS.

EOF
}


#------------------------------------------------------------------------
#   PrintInfo
#------------------------------------------------------------------------


PrintInfo()
{
  cat<<EOF

  System   name     : $SYS_NAME
  Machine  name     : $SYS_ARCH
  Compiler name     : $CXX_NAME
  Compiler command  : $CXX

  Standard    flags : $CXX_STD_FLAGS
  Link        flags : $CXX_LNK_FLAGS
  Debug       flags : $CXX_DBG_FLAGS
  Profiling   flags : $CXX_PRF_FLAGS
  DLL         flags : $CXX_DLL_FLAGS
  Opt level 1 flags : $CXX_OPT_FLAGS1
  Opt level 2 flags : $CXX_OPT_FLAGS2
  Opt level 3 flags : $CXX_OPT_FLAGS3

  Default opt level : $CXX_OPT_LEVEL

EOF
}


#------------------------------------------------------------------------
#   Run
#------------------------------------------------------------------------


Run()
{
  local debug=""
  local profile=""
  local shared=""
  local noExec=""
  local optLevel=0
  local stdFlags=1

  while [ $# -gt 0 ]; do

    case "$1" in

      -h|--help)

        PrintHelp
        exit 0 ;;

      -v|--version)

        PrintVersion
        exit 0 ;;

      -V|--info)

        PrintInfo
        exit 0 ;;

      -n)

        NO_EXEC=1 ;;

      -c)

        COMPILE_ONLY=1 ;;

      -shared)

        shared=1 ;;

      -o)

        shift
        OUTPUT="$1" ;;

      -o?*)

        OUTPUT="${1#-o}" ;;

      -f)

        shift
        FLAGS="$FLAGS $1" ;;

      -f-)

        stdFlags="" ;;

      -f?*)

        FLAGS="$FLAGS ${1#-f}" ;;

      -g)

        debug=1 ;;

      -p)

        profile=1 ;;

      -O)

        optLevel=$CXX_OPT_LEVEL ;;

      -O[0123])

        optLevel="${1#-O}" ;;

      -I)

        shift
        INCDIRS="$INCDIRS $1" ;;

      -I?*)

        INCDIRS="$INCDIRS ${1#-I}" ;;

      -L)

        shift
        LIBDIRS="$LIBDIRS $1" ;;

      -L?*)

        LIBDIRS="$LIBDIRS ${1#-L}" ;;

      -l)

        shift
        LIBS="$LIBS $1" ;;

      -l?*)

        LIBS="$LIBS ${1#-l}" ;;

      --)

        shift
        break ;;

      -*)

        echo "$0: unknown option: $1; try \`$0 --help'." 1>&2
        exit 1 ;;

      *)

        INPUT="$INPUT $1" ;;

    esac

    shift

  done

  INPUT="$INPUT $*"

  if [ -z "$INPUT" ]; then
    echo "$0: no input files." 1>&2
    exit 1
  fi

  if [ -n "$stdFlags" ]; then

    stdFlags="$CXX_STD_FLAGS"

    if [ -n "$debug" ]; then
      stdFlags="$stdFlags $CXX_DBG_FLAGS"
    fi

    if [ -n "$profile" ]; then
      stdFlags="$stdFlags $CXX_PRF_FLAGS"
    fi

    if [ -n "$shared" ]; then
      stdFlags="$stdFlags $CXX_DLL_FLAGS"
    fi

    case $optLevel in

      1)  stdFlags="$stdFlags $CXX_OPT_FLAGS1" ;;
      2)  stdFlags="$stdFlags $CXX_OPT_FLAGS2" ;;
      3)  stdFlags="$stdFlags $CXX_OPT_FLAGS3" ;;

    esac

    if [ -z "$COMPILE_ONLY" ]; then
      stdFlags="$stdFlags $CXX_LNK_FLAGS"
    fi

    FLAGS="$stdFlags $FLAGS"

  fi

  if [ -n "$COMPILE_ONLY" ] || [ -z "$shared" ]; then
    Compile
  else
    MakeDLL
  fi
}


#========================================================================
#   Main script
#========================================================================


Run "$@"
