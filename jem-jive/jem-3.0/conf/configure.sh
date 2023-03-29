
. "scripts/include/setup-${SH_NAME}.sh"


#========================================================================
#   Global variables
#========================================================================


VERBOSE=""
CONF_ARGS="$*"
PROGNAME="configure"

USR_BIT_MODE=""
USR_CXX_FLAGS=""

EXTRA_CONF_DEFS=""
EXTRA_CXX_DEFS=""
EXTRA_MAKE_DEFS=""

USR_JEMDIR="$JEMDIR"

export JEMDIR="$(pwd)"
export JEM_SUBCONF=1


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   CheckJEMDIR ()
#------------------------------------------------------------------------


CheckJEMDIR()
{
  local a=""

  set -- $JEMDIR

  if [ "xx$JEMDIR" != "xx$1" ]; then

    cat<<EOF

The path name of the Jem installation directory seems to contains
spaces and/or special characters. This may result in problems when
compiling Jem-based programs. You are adviced to install Jem in a
directory with a path name that does not contain spaces and/or other
special characters. Alternatively, you can create a link to the Jem
directory and set the JEMDIR environment variable to the name of that
link.

EOF

    printf "Press [RETURN] to continue."
    read a

  fi
}


#------------------------------------------------------------------------
#   ListPackages ()
#------------------------------------------------------------------------


ListPackages()
{
  local list=""
  local dir=""

  for dir in packages/*; do

    if [   -d "$dir" ] && \
       [ ! -e "$dir/DISABLED" ]
    then
      list="$list $(basename "$dir")"
    fi

  done

  echo "$list"
}


#------------------------------------------------------------------------
#   ListPackagesInOrder ()
#------------------------------------------------------------------------


ListPackagesInOrder()
{
  local list=""
  local deps=""
  local p=""
  local i=""

  for p in $(ListPackages); do

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
#   IdentifySystem ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


IdentifySystem()
{
  local sysname=""
  local arg=""

  SYS_NAME=""

  # Check whether a specific system has een specified by the user.

  for arg in "$@" ; do

    case "$arg" in

      --system=?*)

        sysname="${arg#--system=}" ;;

    esac

  done

  if [ -z "$sysname" ] && FindExe --silent uname ; then

    sysname="$(uname -s)"
    sysname="$(ToLower "$sysname")"

  fi

  if [ -n "$sysname" ]; then

    case "$sysname" in

      linux)

        SYS_NAME="linux"  ;;

      darwin)

        SYS_NAME="darwin" ;;

      windows)

        SYS_NAME="windows"
        SYS_IS_POSIX="" ;;

      mingw*_nt*)

        SYS_NAME="windows"
        SYS_IS_POSIX="" ;;

      cygwin*)

        SYS_NAME="cygwin" ;;

    esac

  fi

  if [ -z "$SYS_NAME" ]; then
    Error "Unsupported system -- sorry."
  fi
}


#------------------------------------------------------------------------
#   IdentifyCompiler ()
#------------------------------------------------------------------------


IdentifyCompiler()
{
  local f="$(FindSysconfFile ccident.sh)"

  if [ -z "$f" ]; then
    Oops "Can not find a suitable \`ccident.sh' script."
  fi

  Log "Executing \`$f' ..."

  . "$f"

  if [ -z "$CXX_NAME" ]; then
    Error "Unknown compiler: $CXX."
  fi

  Log "Compiler name:    $CXX_NAME"
  Log "Compiler name2:   $CXX_NAME2"
  Log "Compiler version: $CXX_VERSION"
}


#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  local d=""
  local f=""
  local p=""

  cat<<EOF

Usage: configure [OPTIONS]

Configures the Jem library.

Options:

  -h, --help            : show this help.
  -v, --verbose         : show what is going on.

  --system=SYSTEM       : use SYSTEM as the identifier of the current
                          operating system. This option should only
                          be specified if the operating system can not
                          be identified.
                         (default: automatically detected)

  --cxx=CXX             : use the C++ compiler CXX.
  --cxx-flags=FLAGS     : use FLAGS as the standard compiler fags.
                         (default: depends on compiler and platform)

  -32, -64              : compile in 32-bit or 64-bit mode on
                          platforms that support both 32-bit and
                          64-bit programs.
                         (default: depends on compiler and platform)

  --index-type=TYPE     : use TYPE to define the index type. TYPE can be
                          either \`int' or \`lint' (long integer).
                         (default: --index-type=int)

  --with-threads        : enable  multi-threading support.
  --without-threads     : disable multi-threading support.
                         (default: --with-threads)

  --with-ccache         : use ccache (a fast compiler cache).
  --without-ccache      : do not use ccache.
                         (default: --without-ccache)

  --libs=LIBS           : append LIBS to the list of libraries with
                          which to link.
  --incdirs=DIRS        : search in DIRS for header files not located
                          in the standard compiler include path.
  --libdirs=DIRS        : search in DIRS for libraries not located in
                          the standard compiler search path.

EOF

  # Print system-specific help

  f="$(FindSysconfFile print-help.sh)"

  if [ -n "$f" ]; then
    . "$f"
    echo ""
  fi

  # Print compiler-specific help

  for d in conf/"${SYS_NAME}"-* ; do
    if [ -e "$d/print-help.sh" ]; then
      . "$d/print-help.sh"
      echo ""
    fi
  done

  # Print package-specific help

  for p in $(ListPackages) ; do
    if f="$(FindSysconfFile print-help.sh "packages/$p")"; then
      . "$f"
      echo ""
    fi
  done
}


#------------------------------------------------------------------------
#   PrintArgs ()
#------------------------------------------------------------------------


PrintArgs()
{
  local args=""
  local d=""
  local f=""
  local p=""

  # The main arguments

  args="-h --help -v --verbose\
        --system=\
        --cxx= --cxx-flags= -32 -64\
        --index-type=\
        --with-threads --without-threads\
        --with-ccache  --without-ccache\
        --libs --incdirs --libdirs"

  # The system-specific arguments

  f="$(FindSysconfFile print-args.sh)"

  if [ -n "$f" ]; then
    args="$args $(. "$f")"
  fi

  # The compiler-specific arguments

  for d in conf/"${SYS_NAME}"-* ; do
    if [ -e "$d/print-args.sh" ]; then
      args="$args $(. "$d/print-args.sh")"
    fi
  done

  # The package-specific arguments

  for p in $(ListPackages); do
    f="$(FindSysconfFile print-args.sh packages/$p)"
    if [ -n "$f" ]; then
      args="$args $(. "$f")"
    fi
  done

  printf "%s\n" "$args"
}


#------------------------------------------------------------------------
#   ParseArgs ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


ParseArgs()
{
  local arg=""

  for arg in "$@" ; do

    case "$arg" in

      -h|--help)

        PrintHelp
        exit 0 ;;

      -v|--verbose)

        VERBOSE=1 ;;

      --cxx=?*)

        CXX="${arg#--cxx=}" ;;

      --cxx-flags=?*)

        USR_CXX_FLAGS="${arg#--cxx-flags=}" ;;

      -32)

        USR_BIT_MODE=32 ;;

      -64)

        USR_BIT_MODE=64 ;;

      --index-type=?*)

        INDEX_TYPE="${arg#--index-type=}" ;;

      --with-threads)

        USE_THREADS=1 ;;

      --without-threads)

        USE_THREADS="" ;;

      --with-ccache)

        USE_CCACHE=1 ;;

      --without-ccache)

        USE_CCACHE="" ;;

      --libs=?*)

        SYS_LIBS="$SYS_LIBS ${arg#--libs=}" ;;

      --incdirs=?*)

        SYS_INCDIRS="$SYS_INCDIRS ${arg#--incdirs=}" ;;

      --libdirs=?*)

        SYS_LIBDIRS="$SYS_LIBDIRS ${arg#--libdirs=}" ;;

    esac

  done

  case "$INDEX_TYPE" in

    int|lint) ;;

    *) Error "Invalid index type: $INDEX_TYPE." ;;

  esac
}


#------------------------------------------------------------------------
#   ConfigureSystem ()
#------------------------------------------------------------------------


ConfigureSystem()
{
  local f="$(FindSysconfFile configure.sh)"

  if [ -z "$f" ]; then
    Oops "Unable to configure Jem for this system."
  fi

  Log "Executing \`$f' ..."

  . "$f"

  Log "System architecture: $SYS_ARCH"
}


#------------------------------------------------------------------------
#   ConfigureCompiler ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


ConfigureCompiler()
{
  local f=""

  if [ -z "$CXX_NAME" ]; then
    Oops "Empty compiler name."
  fi

  f="$(FindConfigFile TestCompiler.sh)"

  if [ -z "$f" ]; then
    Error "Unable to configure Jem for the \`$CXX_NAME' compiler."
  fi

  Log "Sourcing \`$f' ..."

  . "$f"

  if [ -n "$USR_CXX_FLAGS" ]; then

    if ! TestCompiler "$CXX" $USR_CXX_FLAGS; then
      Error "Invalid compiler flags: $USR_CXX_FLAGS."
    fi

    CXX_STD_FLAGS="$USR_CXX_FLAGS"

  fi

  f="conf/${SYS_NAME}-${CXX_NAME}/configure.sh"

  if [ ! -e "$f" ]; then
    Error "The \`$CXX_NAME' compiler is not supported on this system."
  fi

  Log "Executing \`$f' ..."

  . "$f"

  if [ -n "$USE_CCACHE" ]; then

    if FindExe --silent ccache ; then
      CXX="ccache $CXX"
      LogEcho "Using ccache to speed up re-compilation."
    else
      USE_CCACHE=""
      LogEcho "Unable to find ccache in \$PATH; disabled."
    fi

  fi
}


#------------------------------------------------------------------------
#   PrintConfigData
#------------------------------------------------------------------------


PrintConfigData()
{
  local var=""
  local val=""

  echo "# Generated on $(date)."
  echo "# CHANGES TO THIS FILE WILL BE LOST."
  echo ""

  # Print the configuration variables

  for var in CONF_ARGS \
             SH \
             SH_NAME \
             AR \
             OBJ_SUFFIX \
             EXE_SUFFIX \
             LIB_PREFIX \
             LIB_SUFFIX \
             DLL_PREFIX \
             DLL_SUFFIX \
             CXX \
             CXX_NAME \
             CXX_VERSION \
             CXX_STD_FLAGS \
             CXX_DBG_FLAGS \
             CXX_PRF_FLAGS \
             CXX_DLL_FLAGS \
             CXX_LNK_FLAGS \
             CXX_BLD_FLAG \
             CXX_OPT_FLAGS1 \
             CXX_OPT_FLAGS2 \
             CXX_OPT_FLAGS3 \
             CXX_OPT_LEVEL \
             SYS_NAME \
             SYS_IS_POSIX \
             SYS_IS_UNIX98 \
             SYS_ARCH \
             SYS_BITS \
             SYS_LIBS \
             SYS_LIBDIRS \
             SYS_INCDIRS \
             INDEX_TYPE \
             USE_THREADS \
             USE_CCACHE
  do
    eval val=\"\$$var\"
    echo "$var=$(quote "$val")"
  done

  echo ""

  printf "$EXTRA_CONF_DEFS"
}


#------------------------------------------------------------------------
#   PrintJemConfig ()
#------------------------------------------------------------------------


PrintJemConfig()
{
  local f="$(FindConfigFile config.h)"

  cat<<EOF

// Generated on $(date).
// CHANGES TO THIS FILE WILL BE LOST.

#ifndef JEM_CONFIG_H
#define JEM_CONFIG_H

#define JEM_OS_$(ToUpper "$SYS_NAME")
#define JEM_ARCH_$(ToUpper "$SYS_ARCH")

EOF

  if [ -n "$USE_THREADS" ]; then
    echo "#define JEM_USE_THREADS"
  fi

  if [ -n "$SYS_IS_POSIX" ]; then
    echo "#define JEM_OS_POSIX"
  fi

  if [ -n "$SYS_IS_UNIX98" ]; then
    echo "#define JEM_OS_UNIX98"
  fi

  printf "$EXTRA_CXX_DEFS"

  if [ -n "$f" ]; then
    cat "$f"
  fi

  echo ""

  if   [ "$INDEX_TYPE" = "int" ]; then
    echo "#define JEM_INDEX_IS_INT"
  else
    echo "#define JEM_INDEX_IS_LINT"
  fi

  echo ""

  if   [ "$INDEX_TYPE" = "int" ]; then
    echo "#define JEM_IDX_T     int"
    echo "#define JEM_UIDX_T    unsigned int"
  else
    echo "#define JEM_IDX_T     long long int"
    echo "#define JEM_UIDX_T    unsigned long long"
  fi

  echo ""

  if   [ "$INDEX_TYPE" = "int" ]; then
    echo "#define JEM_IDX_C(i)  jem::idx_t  ( i )"
    echo "#define JEM_UIDX_C(i) jem::uidx_t ( JEM_CONCAT(i,U) )"
  else
    echo "#define JEM_IDX_C(i)  jem::idx_t  ( JEM_CONCAT(i,LL) )"
    echo "#define JEM_UIDX_C(i) jem::uidx_t ( JEM_CONCAT(i,ULL) )"
  fi

  echo ""
  echo "#endif"
}


#------------------------------------------------------------------------
#   PrintCustomMakedefs ()
#------------------------------------------------------------------------


PrintCustomMakedefs()
{
  # This function can be overloaded to write a custom makefile.

  return 0
}


#------------------------------------------------------------------------
#   PrintMakeConfig
#------------------------------------------------------------------------


PrintMakeConfig()
{
  # Escape the '$' character.

  SYS_LIBS="$(echo "$SYS_LIBS" | sed 's/\$/\$\$/g')"
  SYS_LIBDIRS="$(echo "$SYS_LIBDIRS" | sed 's/\$/\$\$/g')"
  SYS_INCDIRS="$(echo "$SYS_INCDIRS" | sed 's/\$/\$\$/g')"

  cat<<EOF

# Generated on $(date).
# CHANGES TO THIS FILE WILL BE LOST.

ifndef JEM_CONFIG_MK
       JEM_CONFIG_MK = 1

SHELL               = $SH

OBJ_SUFFIX          = $OBJ_SUFFIX
EXE_SUFFIX          = $EXE_SUFFIX
LIB_PREFIX          = $LIB_PREFIX
LIB_SUFFIX          = $LIB_SUFFIX
DLL_PREFIX          = $DLL_PREFIX
DLL_SUFFIX          = $DLL_SUFFIX

SYS_NAME            = $SYS_NAME
SYS_ARCH            = $SYS_ARCH
SYS_BITS            = $SYS_BITS
SYS_LIBS            = $SYS_LIBS
SYS_LIBDIRS         = $SYS_LIBDIRS
SYS_INCDIRS         = $SYS_INCDIRS

CXX                 = $CXX
CXX_NAME            = $CXX_NAME
CXX_VERSION         = $CXX_VERSION

JEM_CXX_STD_FLAGS   = $CXX_STD_FLAGS
JEM_CXX_BLD_FLAGS   = $CXX_BLD_FLAGS
JEM_CXX_LNK_FLAGS   = $CXX_LNK_FLAGS
JEM_CXX_DLL_FLAGS   = $CXX_DLL_FLAGS
JEM_CXX_LIB_FLAGS   = $CXX_LIB_FLAGS
JEM_CXX_DBG_FLAGS   = $CXX_DBG_FLAGS
JEM_CXX_PRF_FLAGS   = $CXX_PRF_FLAGS
JEM_CXX_OPT_FLAGS1  = $CXX_OPT_FLAGS1
JEM_CXX_OPT_FLAGS2  = $CXX_OPT_FLAGS2
JEM_CXX_OPT_FLAGS3  = $CXX_OPT_FLAGS3

JEM_OPT_LEVEL       = $CXX_OPT_LEVEL

EOF

  if [ -n "$EXTRA_MAKE_DEFS" ]; then
    printf "$EXTRA_MAKE_DEFS"
    echo   ""
  fi

  PrintCustomMakedefs

  echo "endif"
}


#------------------------------------------------------------------------
#   PrintShPP ()
#------------------------------------------------------------------------


PrintShPP()
{
  local setup="$JEMDIR/scripts/include/setup-${SH_NAME:?}.sh"

  setup="$(cat "$setup")"

  echo '#!'"$SH"
  echo ""
  echo "$setup"
  echo ""
  echo "SH=$(quote "$SH")"
  echo "SH_NAME=$(quote "$SH_NAME")"
  echo "SH_SETUP=$(quote "$setup")"
  echo ""
  cat  "$JEMDIR/scripts/src/shpp.sh"
}


#------------------------------------------------------------------------
#   InstallScripts ()
#------------------------------------------------------------------------


InstallScripts()
{
  local src=""
  local dest=""
  local f=""

  LogEcho "Installing scripts in \`scripts/bin' ..."

  export TOPDIR='"${JEMDIR:?}"'
  export TOOLKIT="Jem"

  PushDir scripts || Die

  if [ -d bin ]; then
    rm -f -- bin/*
  else
    mkdir bin || Die
  fi

  PrintShPP > bin/shpp

  chmod a+x bin/shpp || Die

  for src in src/*.sh; do

    if [ ! -e "$src" ]; then
      continue
    fi

    dest="${src%.sh}"
    dest="bin/${dest#src/}"

    if [ -e "$dest" ]; then
      continue
    fi

    bin/shpp -x "$src" "$dest" || Die

  done

  PopDir || Die

  if [ ! -d bin ]; then
    mkdir bin || Die
  fi

  PushDir bin || Die

  LogEcho "Installing scripts in \`bin' ..."

  for f in CC \
           cptree \
           mklib \
           objprefix \
           pathto \
           pkgname \
           shpp \
           submake
  do
    cp -f ../scripts/bin/$f $f || Die
  done

  PopDir || Die
}


#------------------------------------------------------------------------
#   ConfigurePackages ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


ConfigurePackages()
{
  local list=""
  local p=""

  echo ""
  echo "Configuring installed packages ..."

  list=""

  for p in $(ListPackagesInOrder); do

    LogEcho ""
    LogEcho "--- package $p ---"
    LogEcho ""

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
#   PrintSummary ()
#------------------------------------------------------------------------


PrintSummary()
{
  local enabled=""
  local disabled=""

  if [ -n "$USE_THREADS" ]; then
    enabled="$enabled threads"
  else
    disabled="$disabled threads"
  fi

  if [ -n "$USE_CCACHE" ]; then
    enabled="$enabled ccache"
  else
    disabled="$disabled ccache"
  fi

  if [ -n "$enabled" ]; then
    echo "  Enabled  options :$enabled"
  fi

  if [ -n "$disabled" ]; then
    echo "  Disabled options :$disabled"
  fi

  if [ -n "$USR_BIT_MODE" ] && \
     [  "xx$USR_BIT_MODE" != "xx$SYS_BITS" ];
  then
    echo "  Ignored  options : -${USR_BIT_MODE}"
  fi

  if [ -n "$CXX_STD_FLAGS" ]; then
    echo "  Compiler flags   : $CXX_STD_FLAGS"
  fi
}


#------------------------------------------------------------------------
#   Run ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


Run()
{
  IdentifySystem "$@"

  CheckArgs "$@"
  ParseArgs "$@"

  CheckJEMDIR

  echo ""
  echo "Using the $SH_NAME shell ($SH)."

  CloseLogfile
  OpenLogfile "$(pwd)/configure.log"

  Log "Arguments:   $@"
  Log "Shell:       $SH"
  Log "System name: $SYS_NAME"

  ConfigureSystem   "$@"
  IdentifyCompiler
  ConfigureCompiler "$@"

  if [ "SYS_BITS" = "32" ]; then
    INDEX_TYPE="int"
  fi

  UnconfigureJem

  WriteFile PrintConfigData  scripts/include/config.sh
  WriteFile PrintJemConfig   include/jem/config.h
  WriteFile PrintMakeConfig  makefiles/config.mk

  InstallScripts
  ConfigurePackages "$@"
  CloseLogfile
}


#========================================================================
#   Main script
#========================================================================


. "$JEMDIR/conf/defaults.sh"
. "$JEMDIR/scripts/include/error.sh"
. "$JEMDIR/scripts/include/utils.sh"
. "$JEMDIR/scripts/include/conf-utils.sh"

Run "$@" || Die

echo ""
echo "Finished the configuration of Jem."
echo ""

PrintSummary

if [ -n "$USR_JEMDIR" ] && [ -d "$USR_JEMDIR" ]; then

  PushDir "$USR_JEMDIR" || Die
  USR_JEMDIR="$(pwd)"
  PopDir                || Die

fi

if [ "xx$JEMDIR" != "xx$USR_JEMDIR" ]; then

  cat<<EOF

Programs based on Jem may use the environment variable JEMDIR
to locate the Jem installation directory. It is a good idea to
set this variable in your shell start-up file. Insert the
following line if you are using a C-like shell (csh, tcsh):

  setenv JEMDIR $(quote "$JEMDIR")

Insert the following if you are using a Bourne-like shell
(ash, bash, ksh, zsh):

  export JEMDIR=$(quote "$JEMDIR")
EOF

fi

cat<<EOF

Now run \`make' to build the Jem libraries. Optionally, run
\`make debug' to build an additional set of libraries with
full debugging support. Note that these libraries require
quite a large amount of disk space.

Finally, run \`make clean' to delete temporary files.

EOF
