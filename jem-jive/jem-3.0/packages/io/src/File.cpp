
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


#include <cstdio>
#include <jem/base/Once.h>
#include <jem/base/CString.h>
#include <jem/base/native/System.h>
#include <jem/util/Flex.h>
#include <jem/io/InputStream.h>
#include <jem/io/OutputStream.h>
#include <jem/io/IOverflowException.h>
#include <jem/io/FileName.h>
#include <jem/io/File.h>
#include <jem/io/native/File.h>
#include <jem/io/native/FileStream.h>
#include <jem/io/native/DirListing.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class File
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int      File::IS_FILE       = 1 << 0;
const int      File::IS_DIR        = 1 << 1;
const int      File::IS_LINK       = 1 << 2;
const int      File::IS_READABLE   = 1 << 3;
const int      File::IS_WRITABLE   = 1 << 4;
const int      File::IS_EXECUTABLE = 1 << 5;

InputStream*   File::in_  = nullptr;
OutputStream*  File::out_ = nullptr;
OutputStream*  File::err_ = nullptr;


//-----------------------------------------------------------------------
//   exists
//-----------------------------------------------------------------------


bool File::exists ( const String& name )
{
  FileInfo  info;

  NativeFile::getInfo ( info, name );

  return (info.mode != 0);
}


//-----------------------------------------------------------------------
//   length
//-----------------------------------------------------------------------


lint File::length ( const String& name )
{
  FileInfo  info;

  NativeFile::getInfo ( info, name );

  if ( ! info.mode )
  {
    String  details = NativeSystem::strerror ();

    throw IOException (
      JEM_FUNC,
      String::format (
        "failed to determine the length of the file `%s\' (%s)",
        name,
        details
      )
    );
  }

  if ( info.length < 0 )
  {
    throw IOverflowException (
      JEM_FUNC,
      "file size too large for a long integer"
    );
  }

  return info.length;
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


FileInfo File::getInfo ( const String& name )
{
  FileInfo  info;

  NativeFile::getInfo ( info, name );

  return info;
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void File::erase ( const String& name )
{
  NativeFile::erase ( name );
}


//-----------------------------------------------------------------------
//   rename
//-----------------------------------------------------------------------


void File::rename

  ( const String&  src,
    const String&  dest )
{
  CString  csrc  ( src );
  CString  cdest ( dest );

  if ( std::rename( csrc.addr(), cdest.addr() ) )
  {
    String  details = NativeSystem::strerror ();

    throw IOException (
      JEM_FUNC,
      String::format (
        "error renaming `%s\' to `%s\' (%s)",
        src,
        dest,
        details
      )
    );
  }
}


//-----------------------------------------------------------------------
//   list
//-----------------------------------------------------------------------


Ref<DirListing> File::list ( const String& name )
{
  return newInstance<NativeDirListing> ( name );
}


//-----------------------------------------------------------------------
//   open
//-----------------------------------------------------------------------


Ref<FileStream> File::open

  ( const String&  name,
    OpenFlags      flags )

{
  return newInstance<NativeFileStream> ( name, flags );
}


//-----------------------------------------------------------------------
//   makeTemp
//-----------------------------------------------------------------------


Ref<FileStream> File::makeTemp

  ( String&    name,
    OpenFlags  flags )

{
  return NativeFile::makeTemp ( name, flags );
}


//-----------------------------------------------------------------------
//   makeDir
//-----------------------------------------------------------------------


void File::makeDir ( const String& name )
{
  NativeFile::makeDir ( name );
}


//-----------------------------------------------------------------------
//   makeDirs
//-----------------------------------------------------------------------


void File::makeDirs ( const String& name )
{
  using jem::util::Flex;

  Flex<String>   stack;
  String         dir;


  if ( name.size() == 0 )
  {
    throw IOException ( JEM_FUNC, "empty directory name" );
  }

  dir = name;

  while ( dir.size() > 0 )
  {
    stack.pushBack( dir );

    dir = FileName::getParent ( dir );
  }

  while ( stack.size() > 0 )
  {
    FileInfo  info;

    NativeFile::getInfo ( info, stack.back() );

    if ( ! (info.mode & IS_DIR) )
    {
      NativeFile::makeDir ( stack.back() );
    }

    stack.popBack ();
  }
}


//-----------------------------------------------------------------------
//   getCwd
//-----------------------------------------------------------------------


String File::getCwd ()
{
  return NativeFile::getCwd ();
}


//-----------------------------------------------------------------------
//   setCwd
//-----------------------------------------------------------------------


void File::setCwd ( const String& path )
{
  NativeFile::setCwd ( path );
}


//-----------------------------------------------------------------------
//   listRoots
//-----------------------------------------------------------------------


Array<String> File::listRoots ()
{
  return NativeFile::listRoots ();
}


//-----------------------------------------------------------------------
//   getStdin
//-----------------------------------------------------------------------


Ref<InputStream> File::getStdin ()
{
  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, init0_ );

  return in_;
}


//-----------------------------------------------------------------------
//   getStdout
//-----------------------------------------------------------------------


Ref<OutputStream> File::getStdout ()
{
  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, init1_ );

  return out_;
}


//-----------------------------------------------------------------------
//   getStderr
//-----------------------------------------------------------------------


Ref<OutputStream> File::getStderr ()
{
  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, init2_ );

  return err_;
}


//-----------------------------------------------------------------------
//   init0_
//-----------------------------------------------------------------------


void File::init0_ ()
{
  Ref<InputStream>  tmp = NativeFile::getStdin  ();

  in_ = tmp.get ();

  Collectable::addRef ( *in_ );
}


//-----------------------------------------------------------------------
//   init1_
//-----------------------------------------------------------------------


void File::init1_ ()
{
  Ref<OutputStream>  tmp = NativeFile::getStdout  ();

  out_ = tmp.get ();

  Collectable::addRef ( *out_ );
}


//-----------------------------------------------------------------------
//   init2_
//-----------------------------------------------------------------------


void File::init2_ ()
{
  Ref<OutputStream>  tmp = NativeFile::getStderr  ();

  err_ = tmp.get ();

  Collectable::addRef ( *err_ );
}


//=======================================================================
//   class FileInfo
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


FileInfo::FileInfo ( const FileInfo& rhs )
{
  mode   = rhs.mode;
  length = rhs.length;
  atime  = rhs.atime;
  mtime  = rhs.mtime;
}


JEM_END_PACKAGE( io )
