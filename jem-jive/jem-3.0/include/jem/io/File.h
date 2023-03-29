
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

#ifndef JEM_IO_FILE_H
#define JEM_IO_FILE_H

#include <jem/base/Time.h>
#include <jem/base/Object.h>
#include <jem/base/array/Array.h>
#include <jem/io/FileFlags.h>


JEM_BEGIN_PACKAGE( io )


class FileInfo;
class DirListing;
class FileStream;
class InputStream;
class OutputStream;


//-----------------------------------------------------------------------
//   class File
//-----------------------------------------------------------------------


class File : public FileFlags
{
 public:

  static const int          IS_FILE;
  static const int          IS_DIR;
  static const int          IS_LINK;
  static const int          IS_READABLE;
  static const int          IS_WRITABLE;
  static const int          IS_EXECUTABLE;


  static bool               exists

    ( const String&           name );

  static lint               length

    ( const String&           name );

  static FileInfo           getInfo

    ( const String&           name );

  static void               erase

    ( const String&           name );

  static void               rename

    ( const String&           src,
      const String&           dest );

  static Ref<DirListing>    list

    ( const String&           name );

  static Ref<FileStream>    open

    ( const String&           name,
      OpenFlags               flags );

  static Ref<FileStream>    makeTemp

    ( String&                 path,
      OpenFlags               flags );

  static void               makeDir

    ( const String&           name );

  static void               makeDirs

    ( const String&           name );

  static String             getCwd        ();

  static void               setCwd

    ( const String&           path );

  static Array<String>      listRoots     ();
  static Ref<InputStream>   getStdin      ();
  static Ref<OutputStream>  getStdout     ();
  static Ref<OutputStream>  getStderr     ();


 private:

  static void               init0_        ();
  static void               init1_        ();
  static void               init2_        ();


 private:

  static InputStream*       in_;
  static OutputStream*      out_;
  static OutputStream*      err_;

};


//-----------------------------------------------------------------------
//   class FileInfo
//-----------------------------------------------------------------------


class FileInfo
{
 public:

  inline                    FileInfo      ();

                            FileInfo

    ( const FileInfo&         rhs );

  inline bool               exists        () const;
  inline bool               isFile        () const;
  inline bool               isDir         () const;
  inline bool               isLink        () const;
  inline bool               isReadable    () const;
  inline bool               isWritable    () const;
  inline bool               isExecutable  () const;


 public:

  int                       mode;
  lint                      length;
  Time                      atime;
  Time                      mtime;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Input>

  inline void               decode

  ( Input&                    in,
    FileInfo&                 info );

template <class Output>

  inline void               encode

  ( Output&                   out,
    const FileInfo&           info );



//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class FileInfo
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline FileInfo::FileInfo ()
{
  mode   = 0;
  length = 0_lint;
}


//-----------------------------------------------------------------------
//   exists
//-----------------------------------------------------------------------


inline bool FileInfo::exists () const
{
  return (mode != 0);
}


//-----------------------------------------------------------------------
//   isFile
//-----------------------------------------------------------------------


inline bool FileInfo::isFile () const
{
  return ((mode & File::IS_FILE) != 0);
}


//-----------------------------------------------------------------------
//   isDir
//-----------------------------------------------------------------------


inline bool FileInfo::isDir () const
{
  return ((mode & File::IS_DIR) != 0);
}


//-----------------------------------------------------------------------
//   isLink
//-----------------------------------------------------------------------


inline bool FileInfo::isLink () const
{
  return ((mode & File::IS_LINK) != 0);
}


//-----------------------------------------------------------------------
//   isReadable
//-----------------------------------------------------------------------


inline bool FileInfo::isReadable () const
{
  return ((mode & File::IS_READABLE) != 0);
}


//-----------------------------------------------------------------------
//   isWritable
//-----------------------------------------------------------------------


inline bool FileInfo::isWritable () const
{
  return ((mode & File::IS_WRITABLE) != 0);
}


//-----------------------------------------------------------------------
//   isExecutable
//-----------------------------------------------------------------------


inline bool FileInfo::isExecutable () const
{
  return ((mode & File::IS_EXECUTABLE) != 0);
}


//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input>

  inline void               decode

  ( Input&                    in,
    FileInfo&                 info )

{
  decode ( in, info.mode, info.length, info.atime, info.mtime );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void               encode

  ( Output&                   out,
    const FileInfo&           info )

{
  encode ( out, info.mode, info.length, info.atime, info.mtime );
}


JEM_END_PACKAGE( io )

#endif
