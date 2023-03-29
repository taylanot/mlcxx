
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
#include <jem/base/Once.h>
#include <jem/base/Time.h>
#include <jem/base/Thread.h>
#include <jem/base/LogBuffer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/SystemException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/native/System.h>
#include <jem/io/FileFlags.h>
#include <jem/io/FileOpenException.h>
#include <jem/io/IOverflowException.h>
#include <jem/io/SyncFailedException.h>
#include <jem/io/CancelledIOException.h>
#include "native/WinFile.h"
#include "native/WinFileStream.h"

extern "C"
{
  #include <windows.h>
}

#include <jem/base/native/winclean.h>


JEM_DEFINE_CLASS( jem::io::WinFileStream );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class WinFileStream
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


WinFileStream*  WinFileStream::stdin_  = nullptr;
WinFileStream*  WinFileStream::stdout_ = nullptr;
WinFileStream*  WinFileStream::stderr_ = nullptr;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


WinFileStream::WinFileStream

  ( void*      hfile,
    OpenFlags  flags )

{
  hfile_ = hfile;
  flags_ = flags;
}


WinFileStream::WinFileStream

  ( const String&  name,
    OpenFlags      flags )

{
  char    path[MAX_PATH + 1];

  DWORD   share  = (FILE_SHARE_DELETE |
                    FILE_SHARE_READ   |
                    FILE_SHARE_WRITE);

  HANDLE  hfile  = INVALID_HANDLE_VALUE;
  DWORD   pause  = 100;
  DWORD   access = 0;
  DWORD   create = 0;
  DWORD   fattr  = FILE_ATTRIBUTE_NORMAL;


  if ( name.size() == 0 )
  {
    throw FileOpenException (
      JEM_FUNC,
      "error opening file: empty file name"
    );
  }

  WinFile::copyName ( path, name );

  if ( flags & FileFlags::READ )
  {
    access |= GENERIC_READ;
    create  = OPEN_EXISTING;
  }

  if ( flags & FileFlags::WRITE )
  {
    access |= GENERIC_WRITE;

    if ( flags & FileFlags::APPEND )
    {
      create = OPEN_ALWAYS;
    }
    else
    {
      create = CREATE_ALWAYS;
    }

    if ( flags & FileFlags::SYNC )
    {
      fattr |= FILE_FLAG_WRITE_THROUGH;
    }
  }

  if ( ! create )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      "invalid file flags (no read/write flags specified)"
    );
  }

  // If the file can not be opened because of a sharing violation,
  // then wait a while and try again.

  for ( int iiter = 0; iiter < 4; iiter++ )
  {
    hfile = CreateFileA ( path, access, share,
                          nullptr, create, fattr, nullptr );

    if ( hfile == INVALID_HANDLE_VALUE &&
         GetLastError() == ERROR_SHARING_VIOLATION )
    {
      Sleep ( pause );

      pause *= 2;
    }
    else
    {
      break;
    }
  }

  if ( hfile != INVALID_HANDLE_VALUE &&
       (flags & FileFlags::APPEND) )
  {
    DWORD  fpos = SetFilePointer ( hfile, 0, nullptr, FILE_END );

    if ( fpos == INVALID_SET_FILE_POINTER &&
         GetLastError() != NO_ERROR )
    {
      CloseHandle ( hfile );

      hfile = INVALID_HANDLE_VALUE;
    }
  }

  if ( hfile == INVALID_HANDLE_VALUE )
  {
    String  details = NativeSystem::strerror ();

    throw FileOpenException (
      JEM_FUNC,
      String::format (
        "error opening file `%s\' (%s)",
        name,
        details
      )
    );
  }

  hfile_ = hfile;
  flags_ = flags;

  LogBuffer::pushBack (
    JEM_FUNC,
    String::format (
      "opened file `%s\' (handle = %p)",
      name,
      (void*) hfile
    )
  );
}


WinFileStream::WinFileStream ( const Self& rhs )
{
  hfile_ = rhs.hfile_;
  flags_ = rhs.flags_;

  if ( flags_ )
  {
    HANDLE  self = GetCurrentProcess ();
    HANDLE  hdup = nullptr;

    if ( ! DuplicateHandle( self, (HANDLE) rhs.hfile_,
                            self, &hdup, 0, FALSE,
                            DUPLICATE_SAME_ACCESS ) )
    {
      String  details = NativeSystem::strerror ();

      throw IOException (
        JEM_FUNC,
        String::format (
          "error duplicating file handle (%s)", details
        )
      );
    }

    hfile_ = (void*) hdup;
  }
}


WinFileStream::~WinFileStream ()
{
  if ( flags_ )
  {
    closeNoThrow ();
  }
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<FileStream> WinFileStream::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   sync
//-----------------------------------------------------------------------


void WinFileStream::sync ()
{
  if ( Thread::cancelled() )
  {
    throw CancelledIOException ( JEM_FUNC, "sync cancelled" );
  }

  if ( ! FlushFileBuffers( (HANDLE) hfile_ ) )
  {
    String  details = NativeSystem::strerror ();

    throw SyncFailedException (
      JEM_FUNC,
      String::format (
        "file synchronization failed (%s)", details
      )
    );
  }
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void WinFileStream::close ()
{
  if ( Thread::cancelled() )
  {
    throw CancelledIOException ( JEM_FUNC, "close cancelled" );
  }

  if ( flags_ )
  {
    HANDLE  hfile = (HANDLE) hfile_;

    if ( ! CloseHandle( hfile ) )
    {
      String  details = NativeSystem::strerror ();

      throw IOException (
        JEM_FUNC,
        String::format (
          "error closing file (%s)", details
        )
      );
    }

    hfile_ = nullptr;
    flags_ = 0;

    LogBuffer::pushBack (
      JEM_FUNC,
      String::format (
        "closed file (handle = %p)",
        (void*) hfile
      )
    );
  }
}


//-----------------------------------------------------------------------
//   closeNoThrow
//-----------------------------------------------------------------------


void WinFileStream::closeNoThrow () noexcept
{
  if ( flags_ )
  {
    HANDLE  hfile = (HANDLE) hfile_;

    CloseHandle ( hfile );

    hfile_ = nullptr;
    flags_ = 0;
  }
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


int WinFileStream::poll ( const Time& timeout )
{
  if ( ! flags_ )
  {
    return 0;
  }

  HANDLE  handle = (HANDLE) hfile_;
  DWORD   ftype  = GetFileType ( handle );
  int     retval = 0;


  if ( ftype == FILE_TYPE_DISK )
  {
    if ( flags_ & FileFlags::READ )
    {
      LARGE_INTEGER  dist;
      LARGE_INTEGER  fpos;
      LARGE_INTEGER  size;

      dist.QuadPart = 0;

      if ( SetFilePointerEx( handle, dist, &fpos, FILE_CURRENT ) )
      {
        if ( GetFileSizeEx( handle, &size ) )
        {
          if ( fpos.QuadPart < size.QuadPart )
          {
            retval |= POLL_READ;
          }
        }
      }
    }

    if ( flags_ & FileFlags::WRITE )
    {
      retval |= POLL_WRITE;
    }
  }

  if ( ftype == FILE_TYPE_CHAR )
  {
    if ( flags_ & FileFlags::READ )
    {
      DWORD  msec;

      if ( timeout < Time::zero() )
      {
        msec = INFINITE;
      }
      else
      {
        msec = (DWORD) (timeout.sec () * 1000L +
                        timeout.nsec() / 1000000L);
      }

      DWORD  result = WaitForSingleObjectEx ( handle, msec, TRUE );

      if ( result == WAIT_OBJECT_0 )
      {
        retval |= POLL_READ;
      }
    }

    if ( flags_ & FileFlags::WRITE )
    {
      retval |= POLL_WRITE;
    }
  }

  if ( ftype == FILE_TYPE_PIPE )
  {
    if ( flags_ & FileFlags::READ )
    {
      DWORD  count;

      if ( PeekNamedPipe( handle, nullptr, 0,
                          nullptr, &count, nullptr ) )
      {
        if ( count > 0 )
        {
          retval |= POLL_READ;
        }
      }
    }
  }

  return retval;
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t WinFileStream::read ( void* buf, idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  DWORD  count;

  if ( (uidx_t) n > maxOf( count ) )
  {
    n = (idx_t) maxOf ( count );
  }

  if ( ! ReadFile( (HANDLE) hfile_, buf,
                   (DWORD) n, &count, nullptr ) )
  {
    throw IOException ( JEM_FUNC, NativeSystem::strerror() );
  }

  if ( Thread::cancelled() )
  {
    throw CancelledIOException ( (idx_t) count,
                                 JEM_FUNC, "read cancelled" );
  }

  return (idx_t) count;
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void WinFileStream::write ( const void* buf, idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  const byte*  src = (const byte*) buf;
  idx_t        k   = 0;


  while ( n > 0 )
  {
    DWORD  count;
    DWORD  len;

    if ( (uidx_t) n > maxOf( len ) )
    {
      len = maxOf ( len );
    }
    else
    {
      len = (DWORD) n;
    }

    if ( ! WriteFile( (HANDLE) hfile_, src,
                      len, &count, nullptr ) )
    {
      throw IOException ( JEM_FUNC, NativeSystem::strerror() );
    }

    k   += (idx_t) count;
    n   -= (idx_t) count;
    src +=         count;

    if ( Thread::cancelled() )
    {
      throw CancelledIOException ( k, JEM_FUNC, "write cancelled" );
    }
  }
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void WinFileStream::writeNoThrow

  ( const void*  buf,
    idx_t        n ) noexcept

{
  const byte*  src = (const byte*) buf;

  while ( n > 0 )
  {
    DWORD  count;
    DWORD  len;

    if ( (uidx_t) n > maxOf( len ) )
    {
      len = maxOf ( len );
    }
    else
    {
      len = (DWORD) n;
    }

    if ( ! WriteFile( (HANDLE) hfile_, src,
                      len, &count, nullptr ) )
    {
      return;
    }

    n   -= (idx_t) count;
    src +=         count;

    if ( Thread::cancelled() )
    {
      return;
    }
  }
}


//-----------------------------------------------------------------------
//   getPosition
//-----------------------------------------------------------------------


lint WinFileStream::getPosition () const
{
  LARGE_INTEGER  fpos;
  LARGE_INTEGER  dist;

  dist.QuadPart = 0;

  if ( ! SetFilePointerEx( (HANDLE) hfile_, dist,
                           &fpos, FILE_CURRENT ) )
  {
    throw IOException ( JEM_FUNC, NativeSystem::strerror() );
  }

  if ( fpos.QuadPart > maxOf<lint>() )
  {
    throw IOverflowException (
      JEM_FUNC,
      "file position too large for a long long integer"
    );
  }

  return (lint) fpos.QuadPart;
}


//-----------------------------------------------------------------------
//   setPosition
//-----------------------------------------------------------------------


void WinFileStream::setPosition ( lint pos )
{
  JEM_PRECHECK ( pos >= 0 );

  LARGE_INTEGER  dist;

  dist.QuadPart = (LONGLONG) pos;

  if ( ! SetFilePointerEx( (HANDLE) hfile_, dist,
                           nullptr, FILE_BEGIN ) )
  {
    throw IOException ( JEM_FUNC, NativeSystem::strerror() );
  }
}


//-----------------------------------------------------------------------
//   getLength
//-----------------------------------------------------------------------


lint WinFileStream::getLength () const
{
  LARGE_INTEGER  size;

  if ( ! GetFileSizeEx( (HANDLE) hfile_, &size ) )
  {
    throw IOException ( JEM_FUNC, NativeSystem::strerror() );
  }

  if ( size.QuadPart > maxOf<lint>() )
  {
    throw IOverflowException (
      JEM_FUNC,
      "file size too large for a long long integer"
    );
  }

  return (lint) size.QuadPart;
}


//-----------------------------------------------------------------------
//   setHandle
//-----------------------------------------------------------------------


void WinFileStream::setHandle

  ( void*      hfile,
    OpenFlags  flags )

{
  if ( flags_ )
  {
    Self::close ();
  }

  hfile_ = hfile;
  flags_ = flags;
}


//-----------------------------------------------------------------------
//   setInherit
//-----------------------------------------------------------------------


void WinFileStream::setInherit ( bool yesno )
{
  if ( flags_ )
  {
    DWORD  flags = 0;

    if ( yesno )
    {
      flags = HANDLE_FLAG_INHERIT;
    }

    if ( ! SetHandleInformation( (HANDLE) hfile_,
                                 HANDLE_FLAG_INHERIT, flags ) )
    {
      String  details = NativeSystem::strerror ();

      throw SystemException (
        JEM_FUNC,
        String::format (
          "error setting file inherit flag (%s)", details
        )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   release
//-----------------------------------------------------------------------


void* WinFileStream::release ()
{
  void*  tmp = hfile_;

  hfile_ = nullptr;
  flags_ = 0;

  return tmp;
}


//-----------------------------------------------------------------------
//   getStdin
//-----------------------------------------------------------------------


Ref<WinFileStream> WinFileStream::getStdin ()
{
  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, initStdIO_, 0 );

  return stdin_;
}


//-----------------------------------------------------------------------
//   getStdout
//-----------------------------------------------------------------------


Ref<WinFileStream> WinFileStream::getStdout ()
{
  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, initStdIO_, 1 );

  return stdout_;
}


//-----------------------------------------------------------------------
//   getStderr
//-----------------------------------------------------------------------


Ref<WinFileStream> WinFileStream::getStderr ()
{
  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, initStdIO_, 2 );

  return stderr_;
}


//-----------------------------------------------------------------------
//   initStdIO_
//-----------------------------------------------------------------------


void WinFileStream::initStdIO_ ( int index )
{
  Self**     stdp = nullptr;

  OpenFlags  flags;
  Ref<Self>  fs;
  DWORD      id;
  HANDLE     hf;


  switch ( index )
  {
  case 0:

    id     = STD_INPUT_HANDLE;
    flags |= FileFlags::READ;
    stdp   = &stdin_;

    break;

  case 1:

    id     = STD_OUTPUT_HANDLE;
    flags |= FileFlags::WRITE;
    stdp   = &stdout_;

    break;

  case 2:

    id     = STD_ERROR_HANDLE;
    flags |= FileFlags::WRITE;
    stdp   = &stderr_;

    break;

  default:

    return;
  }

  hf = GetStdHandle ( id );

  if ( (hf == INVALID_HANDLE_VALUE) || (hf == NULL) )
  {
    flags = 0;
  }

  fs    = newInstance<Self> ( (void*) hf, flags );
  *stdp = fs.get ();

  fs.release ();
}


JEM_END_PACKAGE( io )
