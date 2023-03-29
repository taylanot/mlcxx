
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

#ifndef JEM_IO_READBUFFER_H
#define JEM_IO_READBUFFER_H

#include <jem/io/Reader.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class ReadBuffer
//-----------------------------------------------------------------------


class ReadBuffer
{
 public:

  typedef ReadBuffer      Self;


  explicit                ReadBuffer

    ( idx_t                 size = -1 );

                         ~ReadBuffer  ();

  inline idx_t            size        () const noexcept;
  void                    reset       ();

  inline idx_t            poll

    ( Reader&               in,
      const Time&           timeout );

  inline int              read

    ( Reader&               in );

  idx_t                   read

    ( Reader&               in,
      char*                 buf,
      idx_t                 n );

  idx_t                   readUntil

    ( Reader&               in,
      char*                 buf,
      idx_t                 n,
      int                   delim );

  void                    pushBack

    ( Reader&               in );

  inline void             pushBack

    ( Reader&               in,
      int                   c );

  void                    skipUntil

    ( Reader&               in,
      int                   delim );

  void                    skipWhite

    ( Reader&               in );

  idx_t                   skip

    ( Reader&               in,
      idx_t                 n );

  bool                    scan

    ( Reader&               in,
      const char*           str,
      idx_t                 n );


 private:

                          ReadBuffer  ( const Self& );
  ReadBuffer&             operator =  ( const Self& );

  int                     read_

    ( Reader&               in );

  void                    pushBack_

    ( int                   c );

  void                    fillBuffer_

    ( Reader&               in );

  void                    growStack_  ();


 private:

  static const int        STACK_SIZE_;
  static const char       CANARY_VALUE_;

  char*                   buffer_;
  idx_t                   bufsize_;
  idx_t                   stack_;
  idx_t                   first_;
  idx_t                   last_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t ReadBuffer::size () const noexcept
{
  return (bufsize_ - stack_);
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t ReadBuffer::poll

  ( Reader&      in,
    const Time&  timeout )

{
  idx_t  n  = last_ - first_;

  if ( n > 0 )
  {
    return n;
  }
  else
  {
    return in.poll ( timeout );
  }
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


inline int ReadBuffer::read ( Reader& in )
{
  if ( first_ != last_ )
  {
    return (unsigned char) buffer_[first_++];
  }
  else
  {
    return read_ ( in );
  }
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


inline void ReadBuffer::pushBack ( Reader& in, int c )
{
  if ( c >= 0 )
  {
    if ( first_ > stack_ )
    {
      --first_;
    }
    else
    {
      pushBack_ ( c );
    }
  }
}


JEM_END_PACKAGE( io )

#endif
