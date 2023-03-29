
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

#ifndef JEM_IO_NATIVE_POSIXFILESTREAM_H
#define JEM_IO_NATIVE_POSIXFILESTREAM_H

#include <jem/io/FileFlags.h>
#include <jem/io/FileStream.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class PosixFileStream
//-----------------------------------------------------------------------


class PosixFileStream : public FileStream,
                        public FileFlags
{
 public:

  JEM_DECLARE_CLASS     ( PosixFileStream, FileStream );


  explicit                PosixFileStream

    ( int                   fd    = -1,
      int                   flags =  0 );

  explicit                PosixFileStream

    ( const String&         name,
      OpenFlags             flags = 0 );

                          PosixFileStream

    ( const Self&           rhs );

  virtual
    Ref<FileStream>       dup               () override;

  void                    dup2

    ( int                   newfd );

  virtual void            sync              () override;
  virtual void            close             () override;
  virtual void            closeNoThrow      () noexcept override;

  virtual int             poll

    ( const Time&           timeout )          override;

  virtual idx_t           read

    ( void*                 buf,
      idx_t                 n )                override;

  virtual void            write

    ( const void*           buf,
      idx_t                 n )                override;

  virtual void            writeNoThrow

    ( const void*           buf,
      idx_t                 n )                noexcept override;

  virtual lint            getPosition       () const override;

  virtual void            setPosition

    ( lint                  pos )              override;

  virtual lint            getLength         () const override;

  int                     getFD             () const noexcept;

  void                    setFD

    ( int                   fd,
      int                   flags );

  void                    setCloseOnExec    ();


 protected:

  virtual                ~PosixFileStream   ();


 private:

  int                     fd_;
  int                     flags_;

};


typedef PosixFileStream   NativeFileStream;


JEM_END_PACKAGE( io )

#endif
