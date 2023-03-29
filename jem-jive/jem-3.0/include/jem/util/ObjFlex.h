
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

#ifndef JEM_UTIL_OBJFLEX_H
#define JEM_UTIL_OBJFLEX_H

#include <jem/base/assert.h>
#include <jem/base/Object.h>


namespace jem
{
  namespace io
  {
    class ObjectInput;
    class ObjectOutput;
  }
}


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class ObjFlex
//-----------------------------------------------------------------------


class ObjFlex
{
 public:

                          ObjFlex       () noexcept;

  explicit                ObjFlex

    ( idx_t                 n );

                          ObjFlex

    ( const ObjFlex&        rhs );

                          ObjFlex

    ( ObjFlex&&             rhs )          noexcept;

                         ~ObjFlex       ();

  void                    swap

    ( ObjFlex&              rhs )          noexcept;

  inline void             swap

    ( idx_t                 idx,
      idx_t                 jdx );

  ObjFlex&                operator =

    ( const ObjFlex&        rhs );

  ObjFlex&                operator =

    ( ObjFlex&&             rhs )          noexcept;

  inline Object*          operator []

    ( idx_t                 idx )          const;

  void                    set

    ( idx_t                 idx,
      Object*               obj );

  template <class T>

    inline void           set

    ( idx_t                 idx,
      const Ref<T>&         obj );

  inline Object*          get

    ( idx_t                 idx )          const;

  template <class T>

    inline T*             getAs

    ( idx_t                 idx )          const;

  inline Object*          front         () const;
  inline Object*          back          () const;

  void                    pushBack

    ( Object*               obj );

  template <class T>

    inline void           pushBack

    ( const Ref<T>&         obj );

  void                    popBack       ();

  void                    popBack

    ( idx_t                 n );

  void                    insert

    ( idx_t                 idx,
      Object*               obj  );

  template <class T>

    inline void           insert

    ( idx_t                 idx,
      const Ref<T>&         obj );

  void                    erase

    ( idx_t                 idx );

  void                    erase

    ( idx_t                 idx,
      idx_t                 jdx );

  idx_t                    erase

    ( Object*               obj );

  template <class T>

    inline idx_t           erase

    ( const Ref<T>&         obj );

  void                    clear         ();

  void                    resize

    ( idx_t                 n );

  void                    reserve

    ( idx_t                 cap );

  inline void             trimToSize    ();
  inline idx_t            capacity      () const noexcept;
  inline idx_t            size          () const noexcept;


 private:

  void                    expand_

    ( idx_t                 cap );

  void                    realloc_

    ( idx_t                 cap );


 private:

  idx_t                   size_;
  idx_t                   capacity_;
  Object**                objects_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline void               swap

  ( ObjFlex&                lhs,
    ObjFlex&                rhs ) noexcept;

void                      decode

  ( io::ObjectInput&        in,
    ObjFlex&                ofx );

void                      encode

  ( io::ObjectOutput&       out,
    const ObjFlex&          ofx );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class ObjFlex
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void ObjFlex::swap

  ( idx_t  idx,
    idx_t  jdx )

{
  JEM_ASSERT2 ( checkIndex( idx, size() ) &&
                checkIndex( jdx, size() ),
                "invalid ObjFlex index" );

  Object*  tmp = objects_[idx];

  objects_[idx] = objects_[jdx];
  objects_[jdx] = tmp;
}


//-----------------------------------------------------------------------
//   subscript operator
//-----------------------------------------------------------------------


inline Object* ObjFlex::operator [] ( idx_t idx ) const
{
  JEM_ASSERT2 ( checkIndex( idx, size() ),
                "invalid ObjFlex index" );

  return objects_[idx];
}


//-----------------------------------------------------------------------
//   set
//-----------------------------------------------------------------------


template <class T>

  inline void ObjFlex::set

  ( idx_t          idx,
    const Ref<T>&  obj )

{
  set ( idx, obj.get() );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


inline Object* ObjFlex::get ( idx_t idx ) const
{
  JEM_ASSERT2 ( checkIndex( idx, size() ),
                "invalid ObjFlex index" );

  return objects_[idx];
}


//-----------------------------------------------------------------------
//   getAs
//-----------------------------------------------------------------------


template <class T>

  inline T* ObjFlex::getAs ( idx_t idx ) const

{
  JEM_ASSERT2 ( checkIndex( idx, size() ),
                "invalid ObjFlex index" );

  return static_cast<T*> ( objects_[idx] );
}


//-----------------------------------------------------------------------
//   front & back
//-----------------------------------------------------------------------


inline Object* ObjFlex::front () const
{
  JEM_ASSERT2 ( size() > 0, "empty ObjFlex" );

  return objects_[0];
}


inline Object* ObjFlex::back () const
{
  JEM_ASSERT2 ( size() > 0, "empty ObjFlex" );

  return objects_[size_ - 1];
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


template <class T>

  inline void ObjFlex::pushBack ( const Ref<T>& obj )

{
  pushBack ( obj.get() );
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


template <class T>

  inline void ObjFlex::insert

  ( idx_t          idx,
    const Ref<T>&  obj )

{
  insert ( idx, obj.get() );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------



template <class T>

  inline idx_t ObjFlex::erase ( const Ref<T>& obj )

{
  return erase ( obj.get() );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


inline void ObjFlex::trimToSize ()
{
  realloc_ ( size_ );
}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


inline idx_t ObjFlex::capacity () const noexcept
{
  return capacity_;
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t ObjFlex::size () const noexcept
{
  return size_;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void         swap

  ( ObjFlex&          lhs,
    ObjFlex&          rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( util )

#endif
