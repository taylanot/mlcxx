
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_UTIL_ITEMGROUPSKIN_H
#define JIVE_UTIL_ITEMGROUPSKIN_H

#include <utility>
#include <jive/util/import.h>
#include <jive/util/ItemGroup.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class ItemGroupSkin
//-----------------------------------------------------------------------


class ItemGroupSkin
{
 public:

  typedef ItemGroupSkin     Self;
  typedef ItemGroup         Data;

  inline Event
    < idx_t, Data& >&       newSizeEvent  () const;
  inline Event
    < const Reordering&,
      int, Data& >&         newOrderEvent () const;

  inline void               resetEvents   () const;

  template <class Input>

    inline void             readFrom

    ( Input&                  in );

  template <class Output>

    inline void             writeTo

    ( Output&                 out )          const;

  inline bool               operator ==

    ( const Self&             rhs )          const noexcept;

  inline bool               operator ==

    ( const Nil&              rhs )          const noexcept;

  inline bool               operator ==

    ( std::nullptr_t          rhs )          const noexcept;

  inline bool               operator !=

    ( const Self&             rhs )          const noexcept;

  inline bool               operator !=

    ( const Nil&              rhs )          const noexcept;

  inline bool               operator !=

    ( std::nullptr_t          rhs )          const noexcept;

  explicit inline           operator bool () const noexcept;
  inline bool               operator  !   () const noexcept;

  inline idx_t              size          () const;
  inline IdxVector          getIDs        () const;
  inline IdxVector          getIndices    () const;

  inline idx_t              getIndex

    ( idx_t                   i )            const;

  inline Data*              getData       () const noexcept;

  inline bool               contains

    ( idx_t                   iitem )        const;

  inline void               filter

    ( const IdxVector&        iitems )       const;

  inline idx_t              findMembers

    ( const IdxVector&        jitems,
      const IdxVector&        iitems )       const;

  inline idx_t              findNonMembers

    ( const IdxVector&        jitems,
      const IdxVector&        iitems )       const;

  static inline IdxVector   getUnion

    ( const ItemGroupSkin&    first,
      const ItemGroupSkin&    second );

  static inline IdxVector   getIntersection

    ( const ItemGroupSkin&    first,
      const ItemGroupSkin&    second );

  inline void               store

    ( const String&           name,
      const Properties&       globdat )      const;

  static void               dataError

    ( const String&           context,
      const String&           itemKind,
      const Ref<Data>&        data );


 protected:

  inline                    ItemGroupSkin ()       noexcept;

  explicit inline           ItemGroupSkin

    ( const Ref<Data>&        data )               noexcept;

  explicit inline           ItemGroupSkin

    ( Ref<Data>&&             data )               noexcept;

  inline                    ItemGroupSkin

    ( const Self&             rhs )                noexcept;

  inline                    ItemGroupSkin

    ( Self&&                  rhs )                noexcept;

  inline void               assign_

    ( const Self&             rhs );

  inline void               swap_

    ( Self&                   rhs )                noexcept;


 protected:

  Ref<Data>                 data_;


 private:

  Self&                     operator = ( const Self& );

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Input>

  inline void               decode

  ( Input&                    in,
    ItemGroupSkin&            group );

template <class Output>

  inline void               encode

  ( Output&                   out,
    const ItemGroupSkin&      group );

template <class Output>

  inline void               print

  ( Output&                   out,
    const ItemGroupSkin&      group );

inline bool                 operator ==

  ( const Nil&                lhs,
    const ItemGroupSkin&      rhs )                noexcept;

inline bool                 operator ==

  ( std::nullptr_t            lhs,
    const ItemGroupSkin&      rhs )                noexcept;

inline bool                 operator !=

  ( const Nil&                lhs,
    const ItemGroupSkin&      rhs )                noexcept;

inline bool                 operator !=

  ( std::nullptr_t            lhs,
    const ItemGroupSkin&      rhs )                noexcept;





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class ItemGroupSkin
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline ItemGroupSkin::ItemGroupSkin () noexcept
{}


inline ItemGroupSkin::ItemGroupSkin

  ( const Ref<Data>&  data ) noexcept:

    data_ ( data )

{}


inline ItemGroupSkin::ItemGroupSkin

  ( Ref<Data>&&  data ) noexcept:

    data_ ( std::move( data ) )

{}


inline ItemGroupSkin::ItemGroupSkin ( const Self& rhs ) noexcept:

  data_ ( rhs.data_ )

{}


inline ItemGroupSkin::ItemGroupSkin ( Self&& rhs ) noexcept :

  data_ ( std::move( rhs.data_ ) )

{}


//-----------------------------------------------------------------------
//   newSizeEvent
//-----------------------------------------------------------------------


inline Event < idx_t, ItemGroup& >&

  ItemGroupSkin::newSizeEvent () const

{
  return data_->newSizeEvent;
}


//-----------------------------------------------------------------------
//   newOrderEvent
//-----------------------------------------------------------------------


inline Event < const Reordering&, int, ItemGroup& >&

  ItemGroupSkin::newOrderEvent () const

{
  return data_->newOrderEvent;
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


inline void ItemGroupSkin::resetEvents () const
{
  data_->resetEvents ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


template <class Input>

  inline void ItemGroupSkin::readFrom ( Input& in )

{
  decode ( in, data_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


template <class Output>

  inline void ItemGroupSkin::writeTo ( Output& out ) const

{
  encode ( out, data_ );
}


//-----------------------------------------------------------------------
//   operator ==
//-----------------------------------------------------------------------


inline bool
  ItemGroupSkin::operator == ( const Self& rhs ) const noexcept
{
  return (data_ == rhs.data_);
}


inline bool
  ItemGroupSkin::operator == ( const Nil& rhs ) const noexcept
{
  return (data_ == rhs);
}

inline bool
  ItemGroupSkin::operator == ( std::nullptr_t rhs ) const noexcept
{
  return (data_ == rhs);
}


//-----------------------------------------------------------------------
//   operator !=
//-----------------------------------------------------------------------


inline bool
  ItemGroupSkin::operator != ( const Self& rhs ) const noexcept
{
  return (data_ != rhs.data_);
}


inline bool
  ItemGroupSkin::operator != ( const Nil& rhs ) const noexcept
{
  return (data_ != rhs);
}


inline bool
  ItemGroupSkin::operator != ( std::nullptr_t rhs ) const noexcept
{
  return (data_ != rhs);
}


//-----------------------------------------------------------------------
//   operator bool
//-----------------------------------------------------------------------


inline ItemGroupSkin::operator bool () const noexcept
{
  return (data_ != nullptr);
}


inline bool ItemGroupSkin::operator ! () const noexcept
{
  return (data_ == nullptr);
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t ItemGroupSkin::size () const
{
  return data_->size ();
}


//-----------------------------------------------------------------------
//   getIDs
//-----------------------------------------------------------------------


inline IdxVector ItemGroupSkin::getIDs () const
{
  return data_->getIDs ();
}


//-----------------------------------------------------------------------
//   getIndices
//-----------------------------------------------------------------------


inline IdxVector ItemGroupSkin::getIndices () const
{
  return data_->getIndices ();
}


//-----------------------------------------------------------------------
//   getIndex
//-----------------------------------------------------------------------


inline idx_t ItemGroupSkin::getIndex ( idx_t i ) const
{
  return data_->getIndex ( i );
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


inline ItemGroupSkin::Data* ItemGroupSkin::getData () const noexcept
{
  return data_.get ();
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


inline bool ItemGroupSkin::contains ( idx_t iitem ) const
{
  return data_->contains ( iitem );
}


//-----------------------------------------------------------------------
//   filter
//-----------------------------------------------------------------------


inline void ItemGroupSkin::filter ( const IdxVector& iitems ) const
{
  data_->filter ( iitems );
}


//-----------------------------------------------------------------------
//   findMembers
//-----------------------------------------------------------------------


inline idx_t ItemGroupSkin::findMembers

  ( const IdxVector&  jitems,
    const IdxVector&  iitems ) const

{
  return data_->findMembers ( jitems, iitems );
}


//-----------------------------------------------------------------------
//   findNonMembers
//-----------------------------------------------------------------------


inline idx_t ItemGroupSkin::findNonMembers

  ( const IdxVector&  jitems,
    const IdxVector&  iitems ) const

{
  return data_->findNonMembers ( jitems, iitems );
}


//-----------------------------------------------------------------------
//   getUnion
//-----------------------------------------------------------------------


inline IdxVector ItemGroupSkin::getUnion

  ( const ItemGroupSkin&  first,
    const ItemGroupSkin&  second )

{
  return ItemGroup::getUnion ( * first .getData(),
                               * second.getData() );
}


//-----------------------------------------------------------------------
//   getIntersection
//-----------------------------------------------------------------------


inline IdxVector ItemGroupSkin::getIntersection

  ( const ItemGroupSkin& first,
    const ItemGroupSkin& second )

{
  return ItemGroup::getIntersection ( * first .getData(),
                                      * second.getData() );
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


inline void ItemGroupSkin::store

  ( const String&      name,
    const Properties&  globdat ) const

{
  data_->store ( name, globdat );
}


//-----------------------------------------------------------------------
//   assign_
//-----------------------------------------------------------------------


inline void ItemGroupSkin::assign_ ( const Self& rhs )
{
  data_ = rhs.data_;
}


//-----------------------------------------------------------------------
//   swap_
//-----------------------------------------------------------------------


inline void ItemGroupSkin::swap_ ( Self& rhs ) noexcept
{
  data_.swap ( rhs.data_ );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input>

  inline void             decode

  ( Input&                  in,
    ItemGroupSkin&          group )

{
  group.readFrom ( in );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void             encode

  ( Output&                 out,
    const ItemGroupSkin&    group )

{
  group.writeTo ( out );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output>

  inline void             print

  ( Output&                 out,
    const ItemGroupSkin&    group )

{
  group.getData()->printTo ( out );
}


//-----------------------------------------------------------------------
//   operator ==
//-----------------------------------------------------------------------


inline bool               operator ==

  ( const Nil&              lhs,
    const ItemGroupSkin&    rhs ) noexcept

{
  return rhs.operator == ( lhs );
}


inline bool               operator ==

  ( std::nullptr_t          lhs,
    const ItemGroupSkin&    rhs ) noexcept

{
  return rhs.operator == ( lhs );
}


//-----------------------------------------------------------------------
//   operator !=
//-----------------------------------------------------------------------


inline bool               operator !=

  ( const Nil&              lhs,
    const ItemGroupSkin&    rhs ) noexcept

{
  return rhs.operator != ( lhs );
}


inline bool               operator !=

  ( std::nullptr_t          lhs,
    const ItemGroupSkin&    rhs ) noexcept

{
  return rhs.operator != ( lhs );
}


JIVE_END_PACKAGE( util )

#endif
