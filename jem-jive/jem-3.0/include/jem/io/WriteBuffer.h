
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

#ifndef JEM_IO_WRITEBUFFER_H
#define JEM_IO_WRITEBUFFER_H

#include <jem/io/Writer.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class WriteBuffer
//-----------------------------------------------------------------------


class WriteBuffer
{
 public:

  typedef WriteBuffer     Self;


  explicit                WriteBuffer

    ( idx_t                 size = -1 );

                         ~WriteBuffer ();

  inline idx_t            size        () const noexcept;
  void                    reset       ();

  void                    flush

    ( Writer&               out );

  void                    flushNoThrow

    ( Writer&               out )              noexcept;

  inline void             write

    ( Writer&               out,
      int                   c );

  void                    write

    ( Writer&               out,
      int                   c,
      idx_t                 n );

  void                    write

    ( Writer&               out,
      const char*           buf,
      idx_t                 n );

  void                    writeNoThrow

    ( Writer&               out,
      const char*           buf,
      idx_t                 n )                noexcept;


 private:

                          WriteBuffer ( const Self& rhs );
  WriteBuffer&            operator =  ( const Self& rhs );


 private:

  static const char       CANARY_VALUE_;

  char*                   buffer_;
  idx_t                   bufsize_;
  idx_t                   last_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t WriteBuffer::size () const noexcept
{
  return bufsize_;
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


inline void WriteBuffer::write ( Writer& out, int c )
{
  if ( last_ == bufsize_ )
  {
    flush ( out );
  }

  buffer_[last_++] = (char) c;
}


JEM_END_PACKAGE( io )

#endif
