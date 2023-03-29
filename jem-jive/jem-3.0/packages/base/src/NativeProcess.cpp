
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


#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <jem/base/Error.h>
#include <jem/base/Thread.h>
#include <jem/base/CString.h>
#include <jem/base/MemCache.h>
#include <jem/base/LogBuffer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/SystemException.h>
#include <jem/base/CancelledException.h>
#include <jem/base/native/System.h>
#include <jem/io/IOException.h>
#include <jem/io/FileFlags.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/FileOutputStream.h>
#include <jem/io/FileOpenException.h>
#include <jem/io/native/FileStream.h>
#include "native/PosixProcess.h"

extern "C"
{
  #include <signal.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <sys/wait.h>
  #include <sys/types.h>

#ifdef JEM_OS_DARWIN
  #include <mach-o/dyld.h>
#endif
}


JEM_DEFINE_CLASS( jem::PosixProcess );


JEM_BEGIN_PACKAGE_BASE


using jem::io::FileFlags;
using jem::io::InputStream;
using jem::io::OutputStream;
using jem::io::PosixFileStream;

typedef PosixFileStream  PosixFS;


//=======================================================================
//   private helper functions
//=======================================================================

//-----------------------------------------------------------------------
//   open2_
//-----------------------------------------------------------------------


static void             open2_

  ( Ref<PosixFS>          fs[2],
    const String&         name,
    FileFlags::OpenFlags  flags = 0 )

{
  using jem::io::IOException;
  using jem::io::FileOpenException;

  if ( name.size() == 0 )
  {
    return;
  }

  // Note: the PosixFS objects must be constructed first, so that no
  // resources are leaked when an exception is thrown.

  if ( name == SpawnParams::PIPE )
  {
    int  pipefd[2];

    fs[0] = newInstance<PosixFS> ();
    fs[1] = newInstance<PosixFS> ();

    if ( ::pipe( pipefd ) )
    {
      String  details = NativeSystem::strerror ();

      throw IOException (
        JEM_FUNC,
        String::format (
          "error opening pipe (%s)", details
        )
      );
    }

    fs[0]->setFD ( pipefd[0], FileFlags::READ );
    fs[1]->setFD ( pipefd[1], FileFlags::WRITE );

    fs[0]->setCloseOnExec ();
    fs[1]->setCloseOnExec ();
  }
  else
  {
    int  i = (flags & FileFlags::READ) ? 0 : 1;

    fs[i] = newInstance<PosixFS> ( name.stripWhite(), flags );
  }
}


//=======================================================================
//   class PosixProcess
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PosixProcess::PosixProcess

  ( const Array<String>&  cline,
    const SpawnParams&    params,
    Options               options )

{
  using jem::io::FileInputStream;
  using jem::io::FileOutputStream;

  Ref<PosixFS>  fin [2];
  Ref<PosixFS>  fout[2];
  Ref<PosixFS>  ferr[2];
  Ref<PosixFS>  fsig[2];

  int           len = 0;


  // Open the standard input, output and error files for the child
  // process.

  open2_ ( fin,  params.stdInput,  FileFlags::READ );
  open2_ ( fout, params.stdOutput, FileFlags::WRITE );

  if ( params.stdError == SpawnParams::STD_OUT )
  {
    ferr[0] = fout[0];
    ferr[1] = fout[1];
  }
  else
  {
    open2_ ( ferr, params.stdOutput, FileFlags::WRITE );
  }

  // Open the extra pipe for signalling success/failure of the execve
  // call.

  open2_ ( fsig, SpawnParams::PIPE );

  pid_ = ::fork ();

  if ( pid_ == -1 )
  {
    String  details = NativeSystem::strerror ();

    throw SystemException (
      JEM_FUNC,
      String::format (
        "error creating child process (%s)", details
      )
    );
  }

  if ( pid_ )
  {
    int  fd = fsig[0]->getFD ();

    // This is the parent process. Close the unused file descriptors
    // and check whether the child process has succeeded in calling
    // exec().

    fin [0] = nullptr;
    fout[1] = nullptr;
    ferr[1] = nullptr;
    fsig[1] = nullptr;

    if ( ::read( fd, &len, sizeof(len) ) == sizeof(len) )
    {
      String  what;
      char    buf[256];

      if ( len < 0 )
      {
        len = 0;
      }
      if ( len > (int) sizeof(buf) )
      {
        len = (int) sizeof(buf);
      }

      len = (int) ::read ( fd, buf, (size_t) len );

      if ( len > 0 )
      {
        what = String ( buf, len );
      }
      else
      {
        what = "failed to execute the command line";
      }

      throw SystemException ( JEM_FUNC, what );
    }

    if ( fin[1] )
    {
      stdin_ = newInstance<FileOutputStream> ( fin[1] );
    }

    if ( fout[0] )
    {
      stdout_ = newInstance<FileInputStream> ( fout[0] );
    }

    if      ( ferr[0] == fout[0] )
    {
      stderr_ = stdout_;
    }
    else if ( ferr[0] )
    {
      stderr_ = newInstance<FileInputStream> ( ferr[0] );
    }

    if ( cline.size() > 0 )
    {
      LogBuffer::pushBack (
        JEM_FUNC,
        String::format (
          "started `%s\'",
          cline[0]
        )
      );
    }

    status_ = 0;
  }
  else
  {
    // This is the child process. Close the unused file descriptors.

    fin [1] = nullptr;
    fout[0] = nullptr;
    ferr[0] = nullptr;
    fsig[0] = nullptr;

    if ( fin[0] )
    {
      fin[0]->dup2 ( 0 );
    }

    if ( fout[1] )
    {
      fout[1]->dup2 ( 1 );
    }

    if ( ferr[1] )
    {
      ferr[1]->dup2 ( 2 );
    }

    try
    {
      Self::exec ( cline, options );
    }
    catch ( const Throwable& ex )
    {
      String  what = ex.what        ();
      int     fd   = fsig[1]->getFD ();

      len = (int) what.size ();

      if ( ::write( fd, &len, sizeof(len) ) == sizeof(len) )
      {
        if ( ::write( fd, what.addr(), (size_t) len ) == len )
        {
          // This is to suppress warnings from GCC.

          len = 0;
        }
      }

      ::exit ( 1 );
    }
    catch ( ... )
    {
      int  fd = fsig[1]->getFD ();

      len = 0;

      if ( ::write( fd, &len, sizeof(len) ) == sizeof(len) )
      {
        // This is to suppress warnings from GCC.

        len = 0;
      }

      ::exit ( 1 );
    }
  }
}


PosixProcess::~PosixProcess ()
{
  if ( pid_ )
  {
    ::kill ( pid_, SIGKILL );
  }
}


//-----------------------------------------------------------------------
//   kill
//-----------------------------------------------------------------------


void PosixProcess::kill ()
{
  if ( ! pid_ )
  {
    return;
  }

  if ( ::kill( pid_, SIGTERM ) )
  {
    if ( errno == ESRCH )
    {
      pid_ = 0;
    }
    else
    {
      String  details = NativeSystem::strerror ();

      throw SystemException (
        JEM_FUNC,
        String::format (
          "failed terminating child process %d (%s)",
          pid_,
          details
        )
      );
    }
  }

  wait_ ();
}


//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


int PosixProcess::wait ()
{
  if ( pid_ )
  {
    wait_ ();
  }

  return status_;
}


//-----------------------------------------------------------------------
//   isAlive
//-----------------------------------------------------------------------


bool PosixProcess::isAlive ()
{
  if ( pid_ )
  {
    wait_ ( WNOHANG );
  }

  return (pid_ != 0);
}


//-----------------------------------------------------------------------
//   getStdin
//-----------------------------------------------------------------------


Ref<OutputStream> PosixProcess::getStdin () const
{
  return stdin_;
}


//-----------------------------------------------------------------------
//   getStdout
//-----------------------------------------------------------------------


Ref<InputStream> PosixProcess::getStdout () const
{
  return stdout_;
}


//-----------------------------------------------------------------------
//   getStderr
//-----------------------------------------------------------------------


Ref<InputStream> PosixProcess::getStderr () const
{
  return stderr_;
}


//-----------------------------------------------------------------------
//   exec
//-----------------------------------------------------------------------


void PosixProcess::exec

  ( const Array<String>&  cline,
    Options               options )

{
  const idx_t  argc = cline.size ();

  char*        args;
  char**       argv;
  idx_t        size;


  // Copy the arguments.

  size = (idx_t) (sizeof(char*) * (size_t) (argc + 1));

  for ( idx_t i = 0; i < argc; i++ )
  {
    size += cline[i].size() + 1;
  }

  args  = (char*)  MemCache::alloc ( (size_t) size );
  argv  = (char**) args;

  args += sizeof(char*) * (size_t) (argc + 1);

  for ( idx_t i = 0; i < argc; i++, args++ )
  {
    argv[i] = args;

    std::memcpy ( args, cline[i].addr(), (size_t) cline[i].size() );

    args   += cline[i].size ();
    args[0] = '\0';
  }

  argv[argc] = nullptr;

  ::execvp ( argv[0], argv );

  // An error occurred if this point is reached.

  MemCache::dealloc ( argv, (size_t) size );

  String  details = NativeSystem::strerror ();

  throw SystemException (
    JEM_FUNC,
    String::format (
      "error executing `%s\' (%s)",
      cline[0],
      details
    )
  );
}


//-----------------------------------------------------------------------
//   getExecPath
//-----------------------------------------------------------------------


String PosixProcess::getExecPath ()
{
  String  path;

#if defined(JEM_OS_LINUX)

  char     buf[512];
  ssize_t  n;

  n = ::readlink ( "/proc/self/exe", buf, sizeof(buf) - 1 );

  if ( n > 0 )
  {
    path = String ( buf, (idx_t) n );
  }

#elif defined(JEM_OS_DARWIN)

  char      buf[512];
  uint32_t  len;

  len = sizeof(buf) - 1;

  if ( _NSGetExecutablePath( buf, &len) )
  {
    throw SystemException (
      JEM_FUNC,
      "failed to get the current executable path name; "
      "string buffer is too small"
    );
  }

  path = jem::clone ( buf );

#else

  throw Error (
    JEM_FUNC,
    "function not implemented on this system"
  );

#endif

  return path;
}


//-----------------------------------------------------------------------
//   wait_
//-----------------------------------------------------------------------


int PosixProcess::wait_ ( int flags )
{
  pid_t  result = ::waitpid ( pid_, &status_, flags );

  if ( result == -1 )
  {
    if ( errno == EINTR && Thread::cancelled() )
    {
      throw CancelledException ( JEM_FUNC );
    }
    else
    {
      String  details = NativeSystem::strerror ();

      throw SystemException (
        JEM_FUNC,
        String::format (
          "failed waiting for child process %d (%s)",
          pid_,
          details
        )
      );
    }
  }

  if ( result == 0 )
  {
    return 0;
  }

  pid_ = 0;

  if      ( WIFSIGNALED( status_ ) )
  {
    status_ = KILLED;
  }
  else if ( WIFEXITED( status_ ) )
  {
    status_ = WEXITSTATUS( status_ );
  }


  return status_;
}


JEM_END_PACKAGE_BASE
