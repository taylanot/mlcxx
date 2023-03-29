
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
# DetermineSysBits()
#------------------------------------------------------------------------


DetermineSysBits()
{
  local src="sysbits.cpp"
  local exe="sysbits.exe"
  local obj="sysbits.obj"
  local cmd="$CXX -Fe$exe $src"

  cat<<EOF > "$src"

#include <iostream>

int main ()
{
  std::cout << (int) (8 * sizeof(void*)) << std::flush;
  return 0;
}

EOF

  eval echo '"Executing \`"'"${cmd}"\"\'' ..." >> "$JEM_LOGFILE"'

  eval "$cmd" '>> "$JEM_LOGFILE" 2>&1'

  if [ -x "$exe" ]; then
    SYS_BITS="$("./$exe")"
    rm -f -- "$exe"
  fi

  if [ -z "$SYS_BITS" ]; then
    Error "Failed to determine the size of a pointer."
  fi

  rm -f -- "$src" "$obj"
}


#------------------------------------------------------------------------
#   ConfigureWindowsMSC ()
#------------------------------------------------------------------------


ConfigureWindowsMSC()
{
  local oflags="-DNDEBUG -Oi"

  CXX_LNK_FLAGS="-MD"
  CXX_DBG_FLAGS="-RTCsu -Zi -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=2"
  CXX_PRF_FLAGS=""
  CXX_OPT_LEVEL="3"
  CXX_BLD_FLAGS="-D_CRT_SECURE_NO_WARNINGS -wd4250"
  CXX_OPT_FLAGS1="-O1 $oflags"
  CXX_OPT_FLAGS2="-O2 $oflags"
  CXX_OPT_FLAGS3="-Ox $oflags"

  if [ -z "$USR_CXX_FLAGS" ]; then
    CXX_STD_FLAGS="-W3 -nologo"
  fi

  CXX_STD_FLAGS="$CXX_STD_FLAGS -MD -EHcs"
}


#========================================================================
#   Main script
#========================================================================

echo "Configuring for the Microsoft Visual C++ compiler (cl.exe)."

DetermineSysBits
ConfigureWindowsMSC
