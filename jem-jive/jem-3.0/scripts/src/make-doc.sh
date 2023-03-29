
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

  cd "$TOPDIR/packages" || Die

  for pkg in *; do

    if [ ! -d "$pkg" ] || \
       [   -e "$pkg/DISABLED" ]
    then
      continue
    fi

    if [ -n "$VERBOSE" ]; then
      echo "Processing package $pkg ..."
    fi

    if [ ! -e "$BUILD_DIR/jem/$pkg" ]; then
      mkdir "$BUILD_DIR/jem/$pkg" || Die
    fi

    ProcessPackage "$TOPDIR/packages/$pkg" \
                   "$TOPDIR/include/jem/$pkg" \
                   "$BUILD_DIR/jem/$pkg"

  done

  if [ -e "$JEMDIR/doc/src/main-page.dox" ]; then
    cp -f -- "$JEMDIR/doc/src/main-page.dox" \
             "$BUILD_DIR/main-page.dox"
  fi
}


#========================================================================
#   Main script
#========================================================================


. "${JEMDIR:?}/scripts/include/doctool.sh"
