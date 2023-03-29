
#@set TOPDIR="${TOPDIR:?}"
#@set TOOLKIT="${TOOLKIT:?}"


export TOPDIR


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   CopySourceFile
#------------------------------------------------------------------------


CopySourceFile()
{
  cat "$1" | sed \
    -e '/JEM_DECLARE_CLASS.*/d' \
    -e 's/JEM_BEGIN_PACKAGE_BASE/namespace jem {/' \
    -e 's/JEM_END_PACKAGE_BASE/}/' \
    -e 's/JEM_BEGIN_PACKAGE(\([ a-z0-9]\+\))/\
namespace jem { namespace \\1 {/' \
    -e 's/JEM_END_PACKAGE(\([ a-z0-9]\+\))/} }/' \
    -e 's/JEM_BEGIN_NAMESPACE(\([ a-z0-9]\+\))/\
namespace \\1 {/' \
    -e 's/JEM_END_NAMESPACE(\([ a-z0-9]\+\))//' \
    -e 's/JIVE_BEGIN_PACKAGE(\([ a-z0-9]\+\))/\
namespace jive { namespace \\1 {/' \
    -e 's/JIVE_END_PACKAGE(\([ a-z0-9]\+\))/} }/' \
    > "$2" || return 1
}


#------------------------------------------------------------------------
#   MakeDocTree
#------------------------------------------------------------------------


MakeDocTree()
{
  local pkg=""

  if [ ! -e "$BUILD_DIR/jem" ]; then
    mkdir "$BUILD_DIR/jem" || Die
  fi

  cd "$JEMDIR/packages" || Die

  for pkg in *; do

    if [ ! -d "$pkg" ] || \
       [   -e "$pkg/DISABLED" ]
    then
      continue
    fi

    if [ -n "$VERBOSE" ]; then
      echo "Processing package jem.$pkg ..."
    fi

    if [ ! -e "$BUILD_DIR/jem/$pkg" ]; then
      mkdir "$BUILD_DIR/jem/$pkg" || Die
    fi

    ProcessPackage "$JEMDIR/packages/$pkg" \
                   "$JEMDIR/include/jem/$pkg" \
                   "$BUILD_DIR/jem/$pkg"

  done

  if [ ! -e "$BUILD_DIR/jive" ]; then
    mkdir "$BUILD_DIR/jive" || Die
  fi

  cd "$JIVEDIR/packages" || Die

  for pkg in *; do

    if [ ! -d "$pkg" ] || \
       [   -e "$pkg/DISABLED" ]
    then
      continue
    fi

    if [ -n "$VERBOSE" ]; then
      echo "Processing package jive.$pkg ..."
    fi

    if [ ! -e "$BUILD_DIR/jive/$pkg" ]; then
      mkdir "$BUILD_DIR/jive/$pkg" || Die
    fi

    ProcessPackage "$JIVEDIR/packages/$pkg" \
                   "$JIVEDIR/include/jive/$pkg" \
                   "$BUILD_DIR/jive/$pkg"

  done

  cp -f -- "$JIVEDIR/include/jive/defines.h" \
           "$BUILD_DIR/jive/defines.h"

  if [ -e "$JIVEDIR/doc/src/main-page.dox" ]; then
    cp -f -- "$JIVEDIR/doc/src/main-page.dox" \
             "$BUILD_DIR/main-page.dox"
  fi
}


#========================================================================
#   Main script
#========================================================================


. "${JEMDIR:?}/scripts/include/doctool.sh"
