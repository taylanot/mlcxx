
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


#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/Time.h>
#include <jem/base/Thread.h>
#include <jem/base/CString.h>
#include <jem/base/LogBuffer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/native/System.h>
#include <jem/io/FileFlags.h>
#include <jem/io/FileOpenException.h>
#include <jem/io/IOverflowException.h>
#include <jem/io/SyncFailedException.h>
#include <jem/io/CancelledIOException.h>
#include "native/PosixFileStream.h"

extern "C"
{
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <unistd.h>
  #include <fcntl.h>
}

// Use poll() on Linux and Cygwin, and select() on other
// systems. This is because poll() does not work for standard
// input/output on some systems (such as Mac OS X).

#undef JEM_USE_POLL

#if defined(JEM_OS_LINUX) || defined(JEM_OS_CYGWIN)
#  define JEM_USE_POLL
#endif

#ifdef JEM_USE_POLL

extern "C"
{
  #include <poll.h>
}

#endif


JEM_DEFINE_CLASS( jem::io::PosixFileStream );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class PosixFileStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PosixFileStream::PosixFileStream

  ( int  fd,
    int  flags )

{
  fd_    = fd;
  flags_ = flags;
}


PosixFileStream::PosixFileStream

  ( const String&  name,
    OpenFlags      flags  )

{
  int  ff = 0;
  int  fd;

  if ( name.size() == 0 )
  {
    throw FileOpenException (
      JEM_FUNC,
      "error opening file: empty file name"
    );
  }

  if      ( (flags & FileFlags::READ) &&
            (flags & FileFlags::WRITE) )
  {
    ff = O_RDWR | O_CREAT;
  }
  else if ( flags & FileFlags::READ )
  {
    ff = O_RDONLY;
  }
  else if ( flags & FileFlags::WRITE )
  {
    ff = O_WRONLY | O_CREAT;
  }
  else
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      "invalid file flags (no read/write flags specified)"
    );
  }

  if      ( flags & FileFlags::APPEND )
  {
    ff |= O_APPEND;
  }
  else if ( flags & FileFlags::WRITE )
  {
    ff |= O_TRUNC;
  }

#ifdef O_SYNC

  if ( (flags & FileFlags::WRITE) && (flags & FileFlags::SYNC) )
  {
    ff |= O_SYNC;
  }

#endif

  fd = ::open ( makeCString( name ), ff, 0666 );

  if ( fd == -1 )
  {
    String  details = NativeSystem::strerror ();

    throw FileOpenException (
      JEM_FUNC,
      String::format (
        "error opening file `%s\' (%s)", name, details
      )
    );
  }

#ifdef FD_CLOEXEC

  ::fcntl ( fd, F_SETFD, FD_CLOEXEC );

#endif

  fd_    = fd;
  flags_ = ff;

  LogBuffer::pushBack (
    JEM_FUNC,
    String::format (
      "opened file `%s\' (fd = %d)",
      name,
      fd
    )
  );
}


PosixFileStream::PosixFileStream ( const Self& rhs )
{
  fd_    = rhs.fd_;
  flags_ = rhs.flags_;

  if ( fd_ >= 0 )
  {
    fd_ = ::dup ( rhs.fd_ );

    if ( fd_ < 0 )
    {
      String  details = NativeSystem::strerror ();

      throw IOException (
        JEM_FUNC,
        String::format (
          "error duplicating file descriptor %d (%s)",
          rhs.fd_,
          details
        )
      );
    }
  }
}


PosixFileStream::~PosixFileStream ()
{
  if ( fd_ >= 0 )
  {
    closeNoThrow ();
  }
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<FileStream> PosixFileStream::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   dup2
//-----------------------------------------------------------------------


void PosixFileStream::dup2 ( int newfd )
{
  if ( ::dup2( fd_, newfd ) == -1 )
  {
    String  details = NativeSystem::strerror ();

    throw IOException (
      JEM_FUNC,
      String::format (
        "error duplicating file descriptor %d to %d (%s)",
        fd_,
        newfd,
        details
      )
    );
  }
}


//-----------------------------------------------------------------------
//   sync
//-----------------------------------------------------------------------


void PosixFileStream::sync ()
{
  if ( Thread::cancelled() )
  {
    throw CancelledIOException ( JEM_FUNC, "sync cancelled" );
  }

  if ( ::fsync( fd_ ) != 0 )
  {
    String  details = NativeSystem::strerror ();

    throw SyncFailedException (
      JEM_FUNC,
      String::format ( "file synchronization failed (%s)", details )
    );
  }
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void PosixFileStream::close ()
{
  int  fd = fd_;

  if ( ::close( fd ) != 0 )
  {
    if ( Thread::cancelled() )
    {
      throw CancelledIOException ( JEM_FUNC, "close cancelled" );
    }
    else
    {
      String  details = NativeSystem::strerror ();

      throw IOException (
        JEM_FUNC,
        String::format (
          "error closing file descriptor (%s)", details
        )
      );
    }
  }

  fd_ = -1;

  LogBuffer::pushBack (
    JEM_FUNC,
    String::format (
      "closed file (fd = %d)",
      fd
    )
  );
}


//-----------------------------------------------------------------------
//   closeNoThrow
//-----------------------------------------------------------------------


void PosixFileStream::closeNoThrow () noexcept
{
  if ( fd_ >= 0 )
  {
    ::close ( fd_ );

    fd_ = -1;
  }
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


int PosixFileStream::poll ( const Time& timeout )
{
  if ( fd_ < 0 )
  {
    throw IOException ( JEM_FUNC, "invalid file descriptor" );
  }

  int  result = 0;

#ifdef JEM_USE_POLL

  struct pollfd  pfd;
  int            msec;

  pfd.fd      = fd_;
  pfd.events  = 0;
  pfd.revents = 0;
  pfd.events  = 0;

  if ( flags_ & FileFlags::READ )
  {
    pfd.events |= POLLIN;
  }

  if ( flags_ & FileFlags::WRITE )
  {
    pfd.events |= POLLOUT;
  }

  if ( timeout < Time::zero() )
  {
    msec = -1;
  }
  else
  {
    msec = (int) (timeout.sec () * 1000_lint +
                  timeout.nsec() / 1000000_lint);
  }

  result = ::poll ( &pfd, 1, msec );

  if ( result < 0 )
  {
    throw IOException ( JEM_FUNC, NativeSystem::strerror() );
  }

  if ( result == 1 && (pfd.events & pfd.revents) )
  {
    result = 0;

    if ( pfd.revents & POLLIN )
    {
      result |= POLL_READ;
    }

    if ( pfd.revents & POLLOUT )
    {
      result |= POLL_WRITE;
    }
  }
  else
  {
    result = 0;
  }

#else

  struct timeval  tv;

  fd_set  rfds;
  fd_set  wfds;
  fd_set  xfds;

  FD_ZERO( &rfds );
  FD_ZERO( &wfds );
  FD_ZERO( &xfds );

  FD_SET( fd_, &rfds );

  if ( timeout >= Time::zero() )
  {
    tv.tv_sec  = (time_t)       timeout.sec  ();
    tv.tv_usec = (suseconds_t) (timeout.nsec () / 1000L);

    result = ::select ( fd_ + 1, &rfds, &wfds, &xfds, &tv );
  }
  else
  {
    result = ::select ( fd_ + 1, &rfds, &wfds, &xfds, nullptr );
  }

  if ( result < 0 )
  {
    throw IOException ( JEM_FUNC, NativeSystem::strerror() );
  }

  result = 0;

  if ( FD_ISSET( fd_, &rfds ) )
  {
    result |= POLL_READ;
  }

  if ( FD_ISSET( fd_, &wfds ) )
  {
    result |= POLL_WRITE;
  }

#endif

  return result;
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t PosixFileStream::read ( void* buf, idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  idx_t  k = (idx_t) ::read ( fd_, buf, (size_t) n );

  if ( k < -1 )
  {
    throw IOException ( JEM_FUNC, NativeSystem::strerror() );
  }

  if ( Thread::cancelled() )
  {
    throw CancelledIOException ( k, JEM_FUNC, "read cancelled" );
  }

  return k;
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void PosixFileStream::write ( const void* buf, idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  const byte*  src = (const byte*) buf;
  idx_t        k   = 0;

  while ( n > 0 )
  {
    idx_t  m = (idx_t) ::write ( fd_, src, (size_t) n );

    if ( m == -1 )
    {
      throw IOException ( JEM_FUNC, NativeSystem::strerror() );
    }

    k   += m;
    n   -= m;
    src += m;

    if ( Thread::cancelled() )
    {
      throw CancelledIOException ( k, JEM_FUNC, "write cancelled" );
    }
  }

#ifndef O_SYNC

  if ( flags_ & FileFlags::SYNC )
  {
    sync ();
  }

#endif
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void PosixFileStream::writeNoThrow

  ( const void*  buf,
    idx_t        n ) noexcept

{
  const byte*  src = (const byte*) buf;

  while ( n > 0 )
  {
    idx_t  m = (idx_t) ::write ( fd_, src, (size_t) n );

    if ( m == -1 )
    {
      return;
    }

    n   -= m;
    src += m;

    if ( Thread::cancelled() )
    {
      return;
    }
  }

#ifndef O_SYNC

  if ( flags_ & FileFlags::SYNC )
  {
    sync ();
  }

#endif
}


//-----------------------------------------------------------------------
//   getPosition
//-----------------------------------------------------------------------


lint PosixFileStream::getPosition () const
{
  off_t  pos = ::lseek ( fd_, 0, SEEK_CUR );

  if ( pos == (off_t) -1 )
  {
    throw IOException ( JEM_FUNC, NativeSystem::strerror() );
  }

  if ( pos > maxOf<lint>() )
  {
    throw IOverflowException (
      JEM_FUNC,
      "file position too large for a long long integer"
    );
  }

  return (lint) pos;
}


//-----------------------------------------------------------------------
//   setPosition
//-----------------------------------------------------------------------


void PosixFileStream::setPosition ( lint pos )
{
  JEM_PRECHECK ( pos >= 0 );

  if ( ::lseek( fd_, (off_t) pos, SEEK_SET ) == (off_t) -1 )
  {
    throw IOException ( JEM_FUNC, NativeSystem::strerror() );
  }
}


//-----------------------------------------------------------------------
//   getLength
//-----------------------------------------------------------------------


lint PosixFileStream::getLength () const
{
  struct stat  st;

  if ( ::fstat( fd_, &st ) == -1 )
  {
    throw IOException ( JEM_FUNC, NativeSystem::strerror() );
  }

  if ( st.st_size > maxOf<lint>() )
  {
    throw IOverflowException (
      JEM_FUNC,
      "file size too large for a long long integer"
    );
  }

  return (lint) st.st_size;
}


//-----------------------------------------------------------------------
//   getFD
//-----------------------------------------------------------------------


int PosixFileStream::getFD () const noexcept
{
  return fd_;
}


//-----------------------------------------------------------------------
//   setFD
//-----------------------------------------------------------------------


void PosixFileStream::setFD

  ( int  fd,
    int  flags )

{
  if ( fd_ >= 0 )
  {
    close ();
  }

  fd_    = fd;
  flags_ = flags;
}


//-----------------------------------------------------------------------
//   setCloseOnExec
//-----------------------------------------------------------------------


void PosixFileStream::setCloseOnExec ()
{
#ifdef FD_CLOEXEC

  if ( ::fcntl( fd_, F_SETFD, FD_CLOEXEC ) )
  {
    String  details = NativeSystem::strerror ();

    throw IOException (
      JEM_FUNC,
      String::format (
        "failed to set the close-on-exec flag (%s)", details
      )
    );
  }

#else

  throw SystemException (
    JEM_FUNC,
    "the close-on-exec flag (FD_CLOEXEC) "
    "is not supported by this system"
  );
#endif
}


JEM_END_PACKAGE( io )
