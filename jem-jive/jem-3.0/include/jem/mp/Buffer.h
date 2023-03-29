
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

#ifndef JEM_MP_BUFFER_H
#define JEM_MP_BUFFER_H

#include <jem/mp/Type.h>


JEM_BEGIN_PACKAGE( mp )


//-----------------------------------------------------------------------
//   class Buffer
//-----------------------------------------------------------------------


class Buffer
{
 public:

  inline              Buffer  ()       noexcept;

  inline              Buffer

    ( Type              t,
      void*             a,
      idx_t             n );

  inline              Buffer

    ( Type              t,
      const void*       a,
      idx_t             n );

  inline              Buffer

    ( const Buffer&     rhs )          noexcept;

  inline Type         type    () const noexcept;
  inline idx_t        size    () const noexcept;
  inline void*        addr    () const noexcept;


 private:

  Type                type_;
  idx_t               size_;
  void*               addr_;

};


//-----------------------------------------------------------------------
//   class RecvBuffer
//-----------------------------------------------------------------------


class RecvBuffer : public Buffer
{
 public:

  inline              RecvBuffer ()    noexcept;

  template <class T>
    inline            RecvBuffer

    ( T*                data,
      idx_t             count );

};


//-----------------------------------------------------------------------
//   class SendBuffer
//-----------------------------------------------------------------------


class SendBuffer : public Buffer
{
 public:

  inline              SendBuffer ()    noexcept;

  template <class T>
    inline            SendBuffer

    ( const T*          data,
      idx_t             count );

};





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Buffer
//=======================================================================


inline Buffer::Buffer () noexcept :

  type_ ( NO_TYPE ),
  size_ ( 0_idx ),
  addr_ ( nullptr )

{}


inline Buffer::Buffer ( Type t, void* a, idx_t n ) :

  type_ ( t ),
  size_ ( n ),
  addr_ ( a )

{}


inline Buffer::Buffer ( Type t, const void* a, idx_t n ) :

  type_ ( t ),
  size_ ( n ),
  addr_ ( const_cast<void*>( a ) )

{}


inline Buffer::Buffer ( const Buffer& rhs ) noexcept :

  type_ ( rhs.type_ ),
  size_ ( rhs.size_ ),
  addr_ ( rhs.addr_ )

{}


inline Type Buffer::type () const noexcept
{
  return type_;
}


inline idx_t Buffer::size () const noexcept
{
  return size_;
}


inline void* Buffer::addr () const noexcept
{
  return addr_;
}


//=======================================================================
//   class RecvBuffer
//=======================================================================


inline RecvBuffer::RecvBuffer () noexcept
{}


template <class T>

  inline RecvBuffer::RecvBuffer ( T* data, idx_t count ) :

    Buffer ( getType( (T*) nullptr ), data, count )

{}


//=======================================================================
//   class SendBuffer
//=======================================================================


inline SendBuffer::SendBuffer () noexcept
{}


template <class T>

  inline SendBuffer::SendBuffer ( const T* data, idx_t count ) :

    Buffer ( getType( (T*) nullptr ), data, count )

{}


JEM_END_PACKAGE( mp )

#endif
