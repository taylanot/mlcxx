
#@include "${JEMDIR:?}/scripts/include/config.sh"

#@skip

. "${JEMDIR:?}/scripts/include/config.sh"

#@endskip


#========================================================================
#   Global variables
#========================================================================


VERBOSE=""
PKG_LIST=""
OUTPUT_DIR="."
MERGE_LIBS=""
RELEASE_DIR=""
RELEASE_VERSION=""

export VERBOSE


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: $(basename "$0") [OPTIONS]

Creates a Jive release directory tree.

Options:

  -h, --help        : show this help.
  -v, --verbose     : show what is going on.

  -r VERSION
  --version=VERSION : use VERSION as the release version. If not
                      specified, you will be prompted for it.

  -o DIR
  --output=DIR      : put the Jive release tree in the directory DIR.
                      If not specified, the release tree will be put
                      into the current directory.

  --merge-libs      : merge the package libraries into one library.

EOF
}


#------------------------------------------------------------------------
#   ParseArgs ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


ParseArgs()
{
  while [ $# -gt 0 ]; do

    case "$1" in

      -h|--help)

        PrintHelp
        exit 0 ;;

      -v|--verbose)

        VERBOSE=1 ;;

      -r)

        shift
        RELEASE_VERSION="$1" ;;

      -r?*)

        RELEASE_VERSION="${1#-r}" ;;

      --version=*)

        RELEASE_VERSION="${1#--version=}" ;;

      -o)

        shift
        OUTPUT_DIR="$1" ;;

      -o?*)

        OUTPUT_DIR="${1#-o}" ;;

      --output=*)

        OUTPUT_DIR="${1#--output}" ;;

      --merge-libs)

        MERGE_LIBS=1 ;;

      *)

        echo "$0: invalid option: $1; try \`$0 --help'." 1>&2
        exit 1 ;;

    esac

    shift

  done
}


#------------------------------------------------------------------------
#   MakeDir ( %dir )
#------------------------------------------------------------------------


MakeDir()
{
  if [ -n "$VERBOSE" ]; then
    echo "Creating directory \`$1' ..."
  fi

  mkdir -p -- "$1" || exit 1
}


#------------------------------------------------------------------------
#   WriteFile ( %cmd %file )
#------------------------------------------------------------------------


WriteFile()
{
  if [ -n "$VERBOSE" ]; then
    echo "Writing \`$2' ..."
  fi

  eval "$1" '> "$2"'
}


#------------------------------------------------------------------------
#   ListPackages ()
#------------------------------------------------------------------------


ListPackages()
{
  local curdir="$(pwd)"
  local p=""

  cd "$JIVEDIR/packages" || exit 1

  PKG_LIST=""

  for p in *; do
    if [ -d "$p" ] && [ ! -e "$p"/DISABLED ]; then
      PKG_LIST="$PKG_LIST $p"
    fi
  done

  cd "$curdir" || exit 1
}


#------------------------------------------------------------------------
#   CopyPackages ( [%pkg1 %pkg2 ...] )
#------------------------------------------------------------------------


CopyPackages()
{
  local excl=""
  local src=""
  local dest=""
  local pkg=""
  local i=""

  for pkg in $PKG_LIST; do

    excl=""
    src="$JIVEDIR/packages/$pkg"
    dest="$RELEASE_DIR/packages"

    if [ ! -d "$dest" ]; then
      MakeDir "$dest"
    fi

    if [ -n "$VERBOSE" ]; then
      echo "Copying package \`$pkg' ..."
    fi

    for i in 'OBJ'         \
             'OBJ_OPT'     \
             'OBJ_DBG'     \
             '*.exe'       \
             '*.opt'       \
             '*.dbg'       \
             '*.log'       \
             'core'        \
             'core.[0-9]*' \
             'trash'       \
             'test'
    do
      excl="$excl -x '$i'"
    done

    eval '"$JEMDIR/bin/cptree"' "$excl \"$src\" \"$dest\"" || exit 1

    src="$JIVEDIR/include/jive/$pkg"

    if [ -d "$src" ]; then

      dest="$RELEASE_DIR/include/jive"

      if [ ! -d "$dest" ]; then
        MakeDir "$dest"
      fi

      "$JEMDIR/bin/cptree" "$src" "$dest" || exit 1

    fi

  done
}


#------------------------------------------------------------------------
#   CopyLibs ()
#------------------------------------------------------------------------


CopyLibs()
{
  local list=""
  local cwd="$(pwd)"
  local dir=""
  local lib=""
  local pkg=""
  local i=""

  cd "$JIVEDIR" || exit 1

  if [ -z "$MERGE_LIBS" ]; then

    if [ -n "$VERBOSE" ]; then
      echo "Copying libraries ..."
    fi

    for dir in lib \
               lib-opt \
               lib-dbg
    do
      if [ -d $dir ]; then
        "$JEMDIR/bin/cptree" "$dir" "$RELEASE_DIR" || exit 1
      fi
    done

  else

    if [ -n "$VERBOSE" ]; then
      echo "Creating libraries ..."
    fi

    for dir in lib \
               lib-opt \
               lib-dbg
    do

      if [ ! -d $dir ]; then
        continue
      fi

      list=""

      for pkg in $PKG_LIST; do

        lib="${dir}/${LIB_PREFIX}jive${pkg}${LIB_SUFFIX}"

        if [ -e "$lib" ]; then
          list="$list $lib"
        fi

      done

      if [ -n "$list" ]; then
        lib="${dir}/${LIB_PREFIX}jive${LIB_SUFFIX}"
        "$JEMDIR/bin/mklib" "$RELEASE_DIR/$lib" $list || exit 1
      fi

    done

  fi

  cd "$cwd" || exit 1
}

#------------------------------------------------------------------------
#   AddCopyright ()
#------------------------------------------------------------------------


AddCopyright()
{
  local cfile="$RELEASE_DIR/include/Copyright"

  cat<<EOF > $cfile

/*
 *  Copyright (C) $(date +"%Y") DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: $RELEASE_VERSION
 *  Date:         $(date)
 */

EOF

  if [ -n "$VERBOSE" ]; then
    echo "Adding copyright notice to source files ..."
  fi

  "$JEMDIR/scripts/bin/add-copyright" \
    "$cfile" "$RELEASE_DIR" || exit 1
}


#------------------------------------------------------------------------
#   PrintVersionHeader ()
#------------------------------------------------------------------------


PrintVersionHeader()
{
  cat<<EOF

#ifndef JIVE_VERSION
#define JIVE_VERSION "$RELEASE_VERSION"
#endif

EOF
}


#------------------------------------------------------------------------
#   SetPermissions ()
#------------------------------------------------------------------------


SetPermissions()
{
  if [ -n "$VERBOSE" ]; then
    echo "Updating file permissions ..."
  fi

  chmod -R a+Xr "$RELEASE_DIR"
}


#------------------------------------------------------------------------
#   Run ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


Run()
{
  local dir=""

  if [ -z "$RELEASE_VERSION" ]; then
    printf "Enter the release version: "
    read RELEASE_VERSION
  fi

  RELEASE_DIR="$OUTPUT_DIR/jive-$RELEASE_VERSION"

  if [ -e "$RELEASE_DIR" ]; then
    echo "$0: the file \`$RELEASE_DIR' already exists." 1>&2
    exit 1
  fi

  ListPackages

  echo "Creating release \`$RELEASE_VERSION'" \
       "in directory \`$RELEASE_DIR' ..."

  MakeDir "$RELEASE_DIR"

  for dir in doc \
             doxyfiles \
             images \
             include/jive \
             lib \
             lib-dbg \
             lib-opt \
             makefiles \
             packages \
             examples \
             scripts
  do
    if [ -d $dir ]; then
      MakeDir "$RELEASE_DIR/$dir"
    fi
  done

  if [ -n "$VERBOSE" ]; then
    echo "Copying files ..."
  fi

  cp -- "$JIVEDIR/Makefile" \
        "$JIVEDIR/configure" \
        "$JIVEDIR/LICENSE".* \
        "$RELEASE_DIR" || exit 1

  cp -- "$JIVEDIR/packages/Makefile" \
        "$RELEASE_DIR/packages" || exit 1

  "$JEMDIR/bin/cptree" "$JIVEDIR/scripts/src" \
                       "$RELEASE_DIR/scripts" || exit 1

  "$JEMDIR/bin/cptree" "$JIVEDIR/scripts/conf" \
                       "$RELEASE_DIR/scripts" || exit 1

  "$JEMDIR/bin/cptree" \
    -x "*.dvi" \
    -x "*.log" \
    -x "*.toc" \
    -x "*.aux" \
    "$JIVEDIR/doc" "$RELEASE_DIR" || exit 1

  for dir in doxyfiles \
             images \
             examples \
             makefiles
  do
    "$JEMDIR/bin/cptree" "$JIVEDIR/$dir" "$RELEASE_DIR" || exit 1
  done

  for dir in scripts/include \
             scripts/bin
  do
    "$JEMDIR/bin/cptree" "$JIVEDIR/$dir" \
                         "$RELEASE_DIR/scripts" || exit 1
  done

  cp -- "$JIVEDIR"/include/jive/*.* \
        "$RELEASE_DIR/include/jive" || exit 1

  CopyLibs
  CopyPackages
  AddCopyright

  WriteFile PrintVersionHeader "$RELEASE_DIR/include/jive/version.h"

  SetPermissions

  echo "The release is complete."
}


#========================================================================
#   Main script
#========================================================================


if [ -z "$JEMDIR" ]; then
  echo "$0: undefined variable: JEMDIR."  1>&2
  exit 1
fi

if [ -z "$JIVEDIR" ]; then
  echo "$0: undefined variable: JIVEDIR." 1>&2
  exit 1
fi

ParseArgs "$@"
Run       "$@"
