
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

#ifndef JEM_MP_PORT_H
#define JEM_MP_PORT_H

#include <utility>
#include <jem/mp/Context.h>
#include <jem/mp/Buffer.h>


JEM_BEGIN_PACKAGE( mp )


//-----------------------------------------------------------------------
//   class Port
//-----------------------------------------------------------------------


class Port
{
 public:

  inline                  Port  () noexcept;

  inline                  Port

    ( const Ref<Context>&   ctx,
      int                   rank,
      int                   tag = DEFAULT_TAG );

  template <class T>

    inline void           send

    ( const T&              item );

  template <class T>

    inline void           send

    ( const T*              buf,
      idx_t                 n );

  inline void             send

    ( const SendBuffer&     buf );

  template <class T>

    inline void           recv

    ( T&                    item,
      Status*               stat = 0 );

  template <class T>

    inline void           recv

    ( T*                    buf,
      idx_t                 n,
      Status*               stat = 0 );

  inline void             recv

    ( const RecvBuffer&     buf,
      Status*               stat = 0 );

  template <class T>

    inline Ref<Request>   initSend

    ( const T*              buf,
      idx_t                 n );

  inline Ref<Request>     initSend

    ( const SendBuffer&     buf );

  template <class T>

    inline Ref<Request>   initRecv

    ( T*                    buf,
      idx_t                 n );

  inline Ref<Request>     initRecv

    ( const RecvBuffer&     buf );

  template <class T>

    inline Port&          operator <<

    ( const T&              item );

  template <class T>

    inline Port&          operator >>

    ( T&                    item );



 private:

  Ref<Context>            context_;
  int                     rank_;
  int                     tag_;

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Port
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline Port::Port () noexcept :

  rank_ ( ANY_SOURCE ),
  tag_  (    ANY_TAG )

{}


inline Port::Port

  ( const Ref<Context>&  ctx,
    int                  rank,
    int                  tag ) :

    context_ (  ctx ),
    rank_    ( rank ),
    tag_     (  tag )

{}


//-----------------------------------------------------------------------
//   send
//-----------------------------------------------------------------------


template <class T>

  inline void Port::send ( const T& item )

{
  context_->send ( SendBuffer( &item, 1 ), rank_, tag_ );
}


template <class T>

  inline void Port::send ( const T* buf, idx_t n )

{
  context_->send ( SendBuffer( buf, n ), rank_, tag_ );
}


inline void Port::send ( const SendBuffer& buf )
{
  context_->send ( buf, rank_, tag_ );
}


//-----------------------------------------------------------------------
//   recv
//-----------------------------------------------------------------------


template <class T>

  inline void Port::recv ( T& item, Status* stat )

{
  context_->recv ( RecvBuffer( &item, 1 ), rank_, tag_, stat );
}


template <class T>

  inline void Port::recv ( T* buf, idx_t n, Status* stat )

{
  context_->recv ( RecvBuffer( buf, n ), rank_, tag_, stat );
}


inline void Port::recv ( const RecvBuffer& buf, Status* stat )
{
  context_->recv ( buf, rank_, tag_, stat );
}


//-----------------------------------------------------------------------
//   initSend
//-----------------------------------------------------------------------


template <class T>

  inline Ref<Request> Port::initSend ( const T* buf, idx_t n )

{
  return context_->initSend ( SendBuffer( buf, n ), rank_, tag_ );
}


inline Ref<Request> Port::initSend ( const SendBuffer& buf )
{
  return context_->initSend ( buf, rank_, tag_ );
}


//-----------------------------------------------------------------------
//   initRecv
//-----------------------------------------------------------------------


template <class T>

  inline Ref<Request> Port::initRecv ( T* buf, idx_t n )

{
  return context_->initRecv ( RecvBuffer( buf, n ), rank_, tag_ );
}


inline Ref<Request> Port::initRecv ( const RecvBuffer& buf )
{
  return context_->initRecv ( buf, rank_, tag_ );
}


//-----------------------------------------------------------------------
//   operator <<
//-----------------------------------------------------------------------


template <class T>

  inline Port& Port::operator << ( const T& item )

{
  context_->send ( SendBuffer( &item, 1 ), rank_, tag_ );
  return *this;
}


//-----------------------------------------------------------------------
//   operator >>
//-----------------------------------------------------------------------


template <class T>

  inline Port& Port::operator >> ( T& item )

{
  context_->recv ( RecvBuffer( &item, 1 ), rank_, tag_, 0 );
  return *this;
}


JEM_END_PACKAGE( mp )

#endif
