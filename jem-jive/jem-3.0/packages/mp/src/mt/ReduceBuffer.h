
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

#ifndef JEM_MP_MT_REDUCEBUFFER_H
#define JEM_MP_MT_REDUCEBUFFER_H

#include <cstring>
#include <jem/mp/params.h>
#include <jem/mp/Buffer.h>
#include "error.h"


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( mt )


//-----------------------------------------------------------------------
//   class ReduceBuffer
//-----------------------------------------------------------------------


class ReduceBuffer : public Buffer
{
 public:

  typedef ReduceBuffer  Self;
  typedef Buffer        Super;


  inline                ReduceBuffer () noexcept;

  inline                ReduceBuffer

    ( const RecvBuffer&   buf,
      int                 rnk )         noexcept;

  inline                ReduceBuffer

    ( const SendBuffer&   buf,
      int                 rnk )         noexcept;

  inline int            check

    ( const Buffer&       rhs )         const;

  inline void           copy

    ( const Buffer&       rhs )         const;


 public:

  const int             rank;

};






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline ReduceBuffer::ReduceBuffer () noexcept :

  rank ( -1 )

{}


inline ReduceBuffer::ReduceBuffer

  ( const RecvBuffer& buf, int rnk ) noexcept :

    Super( buf ),
    rank ( rnk )

{}


inline ReduceBuffer::ReduceBuffer

  ( const SendBuffer& buf, int rnk ) noexcept :

    Super( buf ),
    rank ( rnk )

{}


//-----------------------------------------------------------------------
//   check
//-----------------------------------------------------------------------


inline int ReduceBuffer::check ( const Buffer& rhs ) const
{
  if ( type() != rhs.type() )
  {
    return BUFFER_TYPE_ERROR;
  }
  else if ( size() != rhs.size() )
  {
    return BUFFER_SIZE_ERROR;
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   copy
//-----------------------------------------------------------------------


inline void ReduceBuffer::copy ( const Buffer& rhs ) const
{
  std::memcpy ( addr(), rhs.addr(),
                (size_t) (sizeOf( type() ) * size()) );
}


JEM_END_NAMESPACE ( mt )
JEM_END_PACKAGE   ( mp )

#endif
