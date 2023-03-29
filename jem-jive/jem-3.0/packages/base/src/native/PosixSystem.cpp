
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <jem/base/Signals.h>
#include <jem/base/SystemException.h>
#include "native/PosixSystem.h"

extern "C"
{
  #include <signal.h>
  #include <unistd.h>
  #include <sys/utsname.h>

#if defined(JEM_OS_LINUX)
  #include <sys/socket.h>
  #include <sys/types.h>
  #include <sys/ioctl.h>
  #include <net/if.h>
#endif
}

#if defined(__GLIBC__) || defined(__GNU_LIBRARY__)

extern "C"
{
  #include <execinfo.h>
}

#endif


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   non-member functions & data
//=======================================================================


JEM_BEGIN_NAMESPACE( posix )


typedef void  (*SigHandler) ( int );

struct          SigPair
{
  const int       signum;
  SigHandler      handler;
};

static const int         SIG_COUNT  = 4;

static volatile SigPair  sigTable[] =
{
  { Signals::INTERRUPT,    nullptr },
  { Signals::HANGUP,       nullptr },
  { Signals::FP_EXCEPTION, nullptr },
  { Signals::MEMORY_ERROR, nullptr }
};


//-----------------------------------------------------------------------
//   sigGateway
//-----------------------------------------------------------------------


extern "C"
{
  static void sigGateway ( int signo )
  {
    SigHandler  handler = nullptr;
    int         signum  = Signals::NONE;


    switch ( signo )
    {
    case SIGINT:

      signum = Signals::INTERRUPT;
      break;

    case SIGHUP:
    case SIGTERM:

      signum  = Signals::HANGUP;
      break;

    case SIGFPE:

      signum = Signals::FP_EXCEPTION;
      break;

    case SIGSEGV:

      signum = Signals::MEMORY_ERROR;
      break;

    default:

      return;
    }

    for ( int i = 0; i < SIG_COUNT; i++ )
    {
      if ( sigTable[i].signum == signum )
      {
        handler = sigTable[i].handler;
        break;
      }
    }

    if ( handler )
    {
      handler ( signum );
    }
  }
}


JEM_END_NAMESPACE( posix )


//=======================================================================
//   class PosixSystem
//=======================================================================

//-----------------------------------------------------------------------
//   werror
//-----------------------------------------------------------------------


void PosixSystem::werror ( const char* str )
{
  if ( str )
  {
    werror ( str, (idx_t) std::strlen( str ) );
  }
}


void PosixSystem::werror

  ( const char*  str,
    idx_t        len )

{
  while ( len > 0 )
  {
    idx_t  n = (idx_t) ::write ( 2, str, (size_t) len );

    if ( n <= 0 )
    {
      break;
    }

    len -= n;
    str += n;
  }
}


//-----------------------------------------------------------------------
//   strerror
//-----------------------------------------------------------------------


String NativeSystem::strerror ()
{
  String  err;

#if defined(JEM_USE_THREADS)

  char    buf[256];

#if defined(_GNU_SOURCE)

  err = clone ( ::strerror_r( errno, buf, sizeof(buf) ) );

  if ( err.size() )
  {
    idx_t  len = err.size ();

    buf[0] = (char) ::tolower ( err[0] );

    for ( idx_t i = 1; i < len; i++ )
    {
      buf[i] = err[i];
    }

    err = String ( buf, len );
  }

#else

  if ( ::strerror_r( errno, buf, sizeof(buf) ) == 0 )
  {
    buf[0] = (char) ::tolower ( buf[0] );
    err    = clone ( buf );
  }
  else
  {
    err = "unknown error code";
  }

#endif

#else

  err = clone ( ::strerror( errno ) );

#endif

  return err;
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


SystemInfo PosixSystem::getInfo ()
{
  struct utsname  uts;
  SystemInfo      info;
  const char*     str;


  info.totalmem = -1.0;
  info.nrprocs  = -1;

  if ( ::uname( &uts ) == 0 )
  {
    info.osname   = clone ( uts.sysname  );
    info.hostname = clone ( uts.nodename );
    info.machine  = clone ( uts.machine  );
    info.version  = clone ( uts.version  );
  }

  str = ::getenv ( "USER" );

  if ( str )
  {
    info.username = clone ( str );
  }

  str = ::getenv ( "HOME" );

  if ( str )
  {
    info.userhome = clone ( str );
  }

#if ( defined(_SC_PAGESIZE) && defined(_SC_PHYS_PAGES) )

  long  pagesz  = sysconf ( _SC_PAGESIZE );
  long  npages  = sysconf ( _SC_PHYS_PAGES );

  if ( pagesz > 0 && npages > 0 )
  {
    info.totalmem = (double) pagesz * (double) npages;
  }

#endif

#if ( defined(_SC_NPROCESSORS_ONLN) )

  info.nrprocs  = sysconf ( _SC_NPROCESSORS_ONLN );

#endif

  return info;
}


//-----------------------------------------------------------------------
//   getHWAddress
//-----------------------------------------------------------------------


String PosixSystem::getHWAddress ( const String& dev )
{
  String        hwaddr;

#if defined(JEM_OS_LINUX)

  struct ifreq  ifbuf;
  byte*         hwdata;
  int           sockfd;
  idx_t         i, n;


  sockfd = socket ( PF_INET, SOCK_DGRAM, 0 );

  if ( sockfd == -1 )
  {
    goto finally;
  }

  std::memset ( &ifbuf, 0x0, sizeof(ifbuf) );

  n = dev.size ();

  if ( n >= (idx_t) sizeof(ifbuf.ifr_name) )
  {
    goto finally;
  }

  for ( i = 0; i < n; i++ )
  {
    ifbuf.ifr_name[i] = dev[i];
  }

  ifbuf.ifr_name[n] = '\0';

  if ( ioctl( sockfd, SIOCGIFHWADDR, &ifbuf ) )
  {
    goto finally;
  }

  hwdata = (byte*) &ifbuf.ifr_hwaddr.sa_data[0];
  hwaddr = String::format ( "%.2X:%.2X:%.2X:%.2X:%.2X:%.2X",
                            hwdata[0], hwdata[1], hwdata[2],
                            hwdata[3], hwdata[4], hwdata[5] );

finally:

  if ( sockfd >= 0 )
  {
    close ( sockfd );
  }

#endif

  return hwaddr;
}


//-----------------------------------------------------------------------
//   setSigHandler
//-----------------------------------------------------------------------


PosixSystem::SigHandler PosixSystem::setSigHandler

  ( int         signum,
    SigHandler  handler )

{
  struct sigaction  action;

  SigHandler        current = nullptr;
  sigset_t          sigmask;
  int               signo;


  switch ( signum )
  {
  case Signals::INTERRUPT:

    signo = SIGINT;
    break;

  case Signals::HANGUP:

    signo = SIGHUP;
    break;

  case Signals::FP_EXCEPTION:

    signo = SIGFPE;
    break;

  case Signals::MEMORY_ERROR:

    signo = SIGSEGV;
    break;

  default:

    return 0;
  }

  if ( handler )
  {
    action.sa_handler = posix::sigGateway;
  }
  else
  {
    action.sa_handler = SIG_DFL;
  }

  action.sa_flags = 0;

  sigemptyset ( &action.sa_mask );
  sigaddset   ( &action.sa_mask, signo );
  sigprocmask ( SIG_BLOCK, &action.sa_mask, &sigmask );

  for ( int i = 0; i < posix::SIG_COUNT; i++ )
  {
    if ( posix::sigTable[i].signum == signum )
    {
      current = posix::sigTable[i].handler;
      posix::sigTable[i].handler = handler;
      break;
    }
  }

  sigaction   ( signo,       &action,  nullptr );
  sigprocmask ( SIG_SETMASK, &sigmask, nullptr );

  return current;
}


//-----------------------------------------------------------------------
//   getStackTrace / dumpStackTrace
//-----------------------------------------------------------------------


#if defined(__GLIBC__) || defined(__GNU_LIBRARY__)


String PosixSystem::getStackTrace ( int offset )
{
  String   trace;

  char**   names;
  char*    str;

  union
  {
    void*  addr[32];
    char   cbuf[32 * sizeof(void*)];
  };

  int      depth;
  int      i;
  idx_t    j, n;


  if ( offset < 0 )
  {
    return trace;
  }

  depth = ::backtrace         ( addr, 32    );
  names = ::backtrace_symbols ( addr, depth );

  // Skip the stack frame for this function

  offset++;

  n = 0;

  for ( i = offset; i < depth; i++ )
  {
    n += (idx_t) std::strlen ( names[i] );
  }

  if ( (depth - offset) > 0 )
  {
    n += (depth - offset - 1);
  }

  if ( n <= (idx_t) sizeof(cbuf) )
  {
    str = cbuf;
  }
  else
  {
    str = (char*) std::malloc ( (size_t) n );

    if ( ! str )
    {
      std::free ( names );

      return trace;
    }
  }

  j = 0;

  for ( i = offset; i < depth; i++ )
  {
    n = (idx_t) std::strlen ( names[i] );

    if ( ! n )
    {
      continue;
    }

    if ( j )
    {
      str[j++] = '\n';
    }

    std::memcpy ( str + j, names[i], (size_t) n );

    j += n;
  }

  trace = String ( str, j );

  std::free ( names );

  if ( str != cbuf )
  {
    std::free ( str );
  }

  return trace;
}


void PosixSystem::dumpStackTrace ( int offset ) noexcept
{
  void*  addr[32];

  int    depth = ::backtrace ( addr, 32 );

  ::backtrace_symbols_fd ( addr, depth, 2 );
}


#else


String PosixSystem::getStackTrace ( int )
{
  return String ();
}


void PosixSystem::dumpStackTrace ( int ) noexcept
{}


#endif

JEM_END_PACKAGE_BASE
