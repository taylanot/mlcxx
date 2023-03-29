
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
#include <climits>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <jem/ccident.h>
#include <jem/base/Signals.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/SystemException.h>
#include "native/WinSystem.h"

extern "C"
{
  #include <signal.h>
  #include <shlobj.h>
  #include <windows.h>
}

#include "native/winclean.h"


#ifdef JEM_CXX_MSC

#pragma comment( lib, "shell32.lib" )
#pragma comment( lib, "advapi32.lib" )

#endif


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   non-member functions & data
//=======================================================================


JEM_BEGIN_NAMESPACE( win )


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


    // Windows resets the signal handler, so set it again.

    ::signal ( signo, sigGateway );

    switch ( signo )
    {
    case SIGINT:

      signum = Signals::INTERRUPT;
      break;

    case SIGTERM:

      signum = Signals::HANGUP;
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
//   class WinSystem
//=======================================================================

//-----------------------------------------------------------------------
//   werror
//-----------------------------------------------------------------------


void WinSystem::werror ( const char* str )
{
  if ( str )
  {
    werror ( str, (idx_t) std::strlen( str ) );
  }
}


void WinSystem::werror

  ( const char*  str,
    idx_t        len )

{
  HANDLE  fh = GetStdHandle ( STD_ERROR_HANDLE );
  DWORD   sz;
  DWORD   n;

  sz = (DWORD) (((uidx_t) len > UINT_MAX) ? UINT_MAX : len);

  if ( fh )
  {
    WriteFile ( fh, str, sz, &n, nullptr );
  }
}


//-----------------------------------------------------------------------
//   strerror
//-----------------------------------------------------------------------


String NativeSystem::strerror ()
{
  DWORD  ierr = GetLastError ();

  if ( ierr != NO_ERROR )
  {
    char   buf[128];
    DWORD  len;

    len = FormatMessageA ( FORMAT_MESSAGE_FROM_SYSTEM |
                           FORMAT_MESSAGE_IGNORE_INSERTS,
                           nullptr,
                           (DWORD) ierr,
                           0,
                           buf,
                           (DWORD) sizeof(buf),
                           nullptr );

    if ( ! len )
    {
      buf[0] = '\0';
    }
    else
    {
      buf[0] = (char) tolower ( buf[0] );

      while ( len > 0 && std::isspace( buf[len - 1] ) )
      {
        len--;
      }

      if ( len > 0 && buf[len - 1] == '.' )
      {
        len--;
      }

      buf[len] = '\0';
    }

    return clone ( buf );
  }

  if ( errno )
  {
    return clone ( ::strerror( errno ) );
  }
  else
  {
    return "no error";
  }
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


SystemInfo WinSystem::getInfo ()
{
  MEMORYSTATUSEX  memx;
  SYSTEM_INFO     sysi;
  SystemInfo      info;
  char            path[MAX_PATH];
  char            name[256];
  DWORD           size;


  info.osname   = "Windows";
  info.version  = String ();  // Not easy to retrieve anymore.
  info.totalmem = -1.0;
  info.nrprocs  = -1;

  size = (DWORD) sizeof(name);

  if ( GetComputerNameExA( ComputerNamePhysicalDnsHostname,
                           name, &size ) )
  {
    info.hostname = clone ( name );
  }

  size = (DWORD) sizeof(name);

  if ( GetUserNameA( name, &size ) )
  {
    info.username = clone ( name );
  }

  if ( SHGetFolderPathA( nullptr, CSIDL_PERSONAL,
                         nullptr, 0, path ) == S_OK )
  {
    info.userhome = clone ( path );
  }

  GetSystemInfo ( &sysi );

  info.nrprocs = (int) sysi.dwNumberOfProcessors;

  switch ( sysi.wProcessorArchitecture )
  {
  case PROCESSOR_ARCHITECTURE_AMD64:

    info.machine = "x86_64";
    break;

  case PROCESSOR_ARCHITECTURE_IA64:

    info.machine = "ia64";
    break;

  case PROCESSOR_ARCHITECTURE_INTEL:

    info.machine = "x86";
    break;
  }

  std::memset ( &memx, 0x0, sizeof(memx) );

  memx.dwLength = (DWORD) sizeof(memx);

  if ( GlobalMemoryStatusEx( &memx ) )
  {
    info.totalmem = (double) memx.ullTotalPhys;
  }

  return info;
}


//-----------------------------------------------------------------------
//   setSigHandler
//-----------------------------------------------------------------------


WinSystem::SigHandler WinSystem::setSigHandler

  ( int         signum,
    SigHandler  handler )

{
  SigHandler  current = nullptr;
  int         signo;


  switch ( signum )
  {
  case Signals::INTERRUPT:

    signo = SIGINT;
    break;

  case Signals::HANGUP:

    signo = SIGTERM;
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

  for ( int i = 0; i < win::SIG_COUNT; i++ )
  {
    if ( win::sigTable[i].signum == signum )
    {
      current = win::sigTable[i].handler;
      win::sigTable[i].handler = handler;
      break;
    }
  }

  if ( handler )
  {
    ::signal ( signo, win::sigGateway );
  }
  else
  {
    ::signal ( signo, SIG_DFL );
  }

  return current;
}


//-----------------------------------------------------------------------
//   getStackTrace / dumpStackTrace
//-----------------------------------------------------------------------


String WinSystem::getStackTrace ( int offset )
{
  return String ();
}


void WinSystem::dumpStackTrace ( int offset ) noexcept
{}


JEM_END_PACKAGE_BASE
