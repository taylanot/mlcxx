
SYS_IS_POSIX=1

if [ -n "$USE_THREADS" ]; then
  SYS_LIBS="$SYS_LIBS pthread"
fi

SYS_LIBS="$SYS_LIBS m"
