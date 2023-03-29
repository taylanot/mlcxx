
#@include "${JEMDIR:?}/scripts/include/error.sh"
#@include "${JEMDIR:?}/scripts/include/utils.sh"

#@skip

if [ -n "$JEM_SCRIPTS_DOCTOOL_SH" ]; then
  return 0
else
  JEM_SCRIPTS_DOCTOOL_SH=1
fi

. "${JEMDIR:?}/scripts/include/error.sh"
. "${JEMDIR:?}/scripts/include/utils.sh"

#@endskip


#========================================================================
#   Global variables
#========================================================================


VERBOSE=""
BUILD_DIR="$TOPDIR/doxyfiles/build"


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

Creates the $TOOLKIT reference manual.

Options:

  -h, --help    : show this help.
  -v, --verbose : show what is going on.

EOF
}


#------------------------------------------------------------------------
#   ParseArgs ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


ParseArgs()
{
  local arg=""

  for arg in "$@"; do

    case "$arg" in

      -h|--help)

        PrintHelp
        exit 0 ;;

      -v|--verbose)

        VERBOSE=1 ;;

      *)

        OptionError "$arg" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   PrintExamplesPage ()
#------------------------------------------------------------------------


PrintExamplesPage()
{
  local f=""

  for f in *.cpp ; do

    if [ -e "$f" ]; then
      echo "/*!"
      echo "  \\\\example $f"
      echo "*/"
      echo ""
    fi

  done
}


#------------------------------------------------------------------------
#   CopyFiles ( %srcDir, %destDir )
#------------------------------------------------------------------------


CopyFiles()
{
  local curDir="$(pwd)"
  local srcDir="$1"
  local destDir="$2"
  local f=""

  for f in * ; do

    if [ ! -e "$f" ]; then
      continue
    fi

    if [ -d "$f" ] && [ -d "$srcDir/$f" ]; then

      if [ ! -e "$destDir/$f" ]; then
        mkdir "$destDir/$f" || return 1
      fi

      cd "$f" || return 1

      CopyFiles "$srcDir/$f" "$destDir/$f" || return 1

      cd "$curDir" || return 1

      continue

    fi

    case "$f" in

      *.h)

        if [ -e "$srcDir/$f" ]; then
          CopySourceFile "$srcDir/$f" "$destDir/$f"
        fi
        ;;

      *.dox)

        CopySourceFile "$f" "$destDir/$f" || return 1
        ;;

      *)
        ;;

    esac

  done
}


#------------------------------------------------------------------------
#   ProcessPackage ( %pkgDir, %incDir, %destDir )
#------------------------------------------------------------------------


ProcessPackage()
{
  local curDir="$(pwd)"
  local pkgDir="$1"
  local incDir="$2"
  local destDir="$3"
  local f=""

  if [ ! -e "$pkgDir/doc" ]; then
    return 0
  fi

  if [ ! -e "$destDir" ]; then
    mkdir -p "$destDir" || return 1
  fi

  if [ -e "$pkgDir/doc/src" ]; then

    cd "$pkgDir/doc/src" || return 1

    CopyFiles "$incDir" "$destDir"

    cd "$curDir" || return 1

  fi

  if [ ! -e "$BUILD_DIR/images" ]; then
    mkdir "$BUILD_DIR/images" || return 1
  fi

  if [ -e "$pkgDir/doc/images" ]; then

    cd "$pkgDir/doc/images" || return 1

    for f in * ; do

      if [ ! -e "$f" ]; then
        continue
      fi

      if [ -e "$BUILD_DIR/images/$f" ]; then
        echo "*** Warning: duplicate image: $f" 1>&2
      fi

      cp -f -- "$f" "$BUILD_DIR/images/$f"

    done

    cd "$curDir" || return 1

  fi

  if [ -e "$pkgDir/examples" ]; then

    if [ ! -e "$BUILD_DIR/examples" ]; then
      mkdir "$BUILD_DIR/examples" || return 1
    fi

    cd "$pkgDir/examples" || return 1

    for f in *.cpp ; do

      if [ ! -e "$f" ]; then
        continue
      fi

      if [ -e "$BUILD_DIR/examples/$f" ]; then
        echo "*** Warning: duplicate example: $f" 1>&2
      fi

      cp -f -- "$f" "$BUILD_DIR/examples/$f"

    done

    cd "$curDir" || return 1

  fi
}


#------------------------------------------------------------------------
#   Run ()
#------------------------------------------------------------------------


Run()
{
  local f=""

  cd "$TOPDIR/doxyfiles" || Die

  if [ -e build ]; then
    rm -f -r build
  fi

  mkdir build || Die

  if [ -n "$VERBOSE" ]; then
    echo "Creating the Doxygen configuration file ..."
  fi

  echo "# Generated on $(date)."              >  build/Doxyfile
  echo "# CHANGES TO THIS FILE WILL BE LOST." >> build/Doxyfile
  echo ""                                     >> build/Doxyfile

  if [ -e doxygen.conf ]; then
    "$JEMDIR/scripts/bin/shpp" doxygen.conf   >> build/Doxyfile
  fi

  cd build || Die

  if [ -n "$VERBOSE" ]; then
    echo "Creating the documentation source tree ..."
  fi

  MakeDocTree

  cd "$TOPDIR/doxyfiles/build" || Die

  if [ -d "examples" ]; then

    cd "examples" || Die

    if [ -n "$VERBOSE" ]; then
      echo "Creating the examples page ..."
    fi

    PrintExamplesPage > examples.dox

    cd .. || Die

  fi

  if [ -n "$VERBOSE" ]; then
    echo "Creating the Doxygen header/footer/style files ..."
  fi

  touch header.html
  touch footer.html
  touch style.css

  doxygen -w html header.html \
                  footer.html \
                  style.css   \
                  Doxyfile

  doxygen -l      layout.xml

  mv -f header.html default-header.html || Die
  mv -f footer.html default-footer.html || Die
  mv -f style.css   default-style.css   || Die
  mv -f layout.xml  default-layout.xml  || Die

  for f in header.html \
           footer.html \
           layout.xml  \
           style.css; do
    if [ -e "../$f" ]; then
      "$JEMDIR/scripts/bin/shpp" "../$f" "$f"
    fi
  done

  if [ -n "$VERBOSE" ]; then
    echo "Running Doxygen ..."
  fi

  doxygen

  cd "$TOPDIR/doxyfiles" || Die

  if [ -e logo.png ] && [ -d "$TOPDIR/doc/html" ]; then
    cp -f logo.png "$TOPDIR/doc/html"
  fi

  if [ -n "$VERBOSE" ]; then
    echo "Cleaning up build directory ..."
  fi

  rm -f -r -- build
}


#========================================================================
#   Main script
#========================================================================


ParseArgs "$@"
Run

