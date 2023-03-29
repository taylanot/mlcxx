
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

#ifndef JIVE_MP_BORDERSET_H
#define JIVE_MP_BORDERSET_H

#include <jive/util/GroupSet.h>
#include <jive/util/ItemSetSkin.h>
#include <jive/mp/import.h>


JIVE_BEGIN_PACKAGE( mp )


using jem::util::Event;


//-----------------------------------------------------------------------
//   enum BorderKind
//-----------------------------------------------------------------------


enum BorderKind
{
  RECV_BORDERS      = 1,
  SEND_BORDERS      = 2,
  SEND_RECV_BORDERS = 3
};


//-----------------------------------------------------------------------
//   class BorderSet
//-----------------------------------------------------------------------


class BorderSet : public ItemSetSkin
{
 public:

  typedef BorderSet         Self;
  typedef ItemSetSkin       Super;

  typedef util::GroupSet    Data;
  typedef util::Topology    Topology;


  inline                    BorderSet       ()       noexcept;

  explicit inline           BorderSet

    ( const Ref<Data>&        borders )              noexcept;

  explicit inline           BorderSet

    ( Ref<Data>&&             borders )              noexcept;

  inline Event
    < Data& >&              newTopoEvent    () const;
  inline Data*              getData         () const noexcept;
  inline ItemSet*           getItems        () const;

  inline String             getBorderPath

    ( idx_t                   borderID )       const;

  inline idx_t              findBorder

    ( idx_t                   borderID )       const;

  inline idx_t              getBorderID

    ( idx_t                   iborder )        const;

  inline idx_t              maxBorderSize   () const;

  inline idx_t              getBorderSize

    ( idx_t                   iborder )        const;

  inline idx_t              getBorderItems

    ( const IdxVector&        iitems,
      idx_t                   iborder )        const;

  idx_t                     getBorderItemIDs

    ( const IdxVector&        itemIDs,
      idx_t                   iborder )        const;

  inline void               getSomeBorderItems

    ( const IdxVector&        iitems,
      const IdxVector&        index,
      idx_t                   iborder )        const;

  inline IdxVector          toVector        () const;
  inline Topology           toMatrix        () const;

  inline IdxVector          getMembersOf

    ( const IdxVector&        iborders )       const;

  inline IdxVector          getUniqueMembersOf

    ( const IdxVector&        iborders )       const;

  void                      store

    ( BorderKind              kind,
      const Properties&       globdat )        const;

  static Self               find

    ( BorderKind              kind,
      const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Self               get

    ( BorderKind              kind,
      const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static bool               hasBorders

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );


 protected:

  inline Data&              borders_        () const;

};





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class BorderSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline BorderSet::BorderSet () noexcept
{}


inline BorderSet::BorderSet

  ( const Ref<Data>&  borders ) noexcept :

    Super ( borders )

{}


inline BorderSet::BorderSet

  ( Ref<Data>&&  borders ) noexcept :

    Super ( std::move( borders ) )

{}


//-----------------------------------------------------------------------
//   newTopoEvent
//-----------------------------------------------------------------------


inline Event < util::GroupSet& >& BorderSet::newTopoEvent () const
{
  return borders_().newTopoEvent;
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


inline BorderSet::Data* BorderSet::getData () const noexcept
{
  return jem::staticCast<Data*> ( data_ );
}


//-----------------------------------------------------------------------
//   getNodes
//-----------------------------------------------------------------------


inline ItemSet* BorderSet::getItems () const
{
  return borders_().getGroupedItems ();
}


//-----------------------------------------------------------------------
//   getBorderPath
//-----------------------------------------------------------------------


inline String BorderSet::getBorderPath ( idx_t borderID ) const
{
  return borders_().getItemPath ( borderID );
}


//-----------------------------------------------------------------------
//   findBorder
//-----------------------------------------------------------------------


inline idx_t BorderSet::findBorder ( idx_t borderID ) const
{
  return borders_().findItem ( borderID );
}


//-----------------------------------------------------------------------
//   getBorderID
//-----------------------------------------------------------------------


inline idx_t BorderSet::getBorderID ( idx_t iborder  ) const
{
  return borders_().getItemID ( iborder );
}


//-----------------------------------------------------------------------
//   maxBorderSize
//-----------------------------------------------------------------------


inline idx_t BorderSet::maxBorderSize () const
{
  return borders_().maxGroupSize ();
}


//-----------------------------------------------------------------------
//   getBorderSize
//-----------------------------------------------------------------------


inline idx_t BorderSet::getBorderSize ( idx_t iborder ) const
{
  return borders_().getGroupSize ( iborder );
}


//-----------------------------------------------------------------------
//   getBorderItems
//-----------------------------------------------------------------------


inline idx_t BorderSet::getBorderItems

  ( const IdxVector&  iitems,
    idx_t               iborder ) const

{
  return borders_().getGroupMembers ( iitems, iborder );
}


//-----------------------------------------------------------------------
//   getSomeBorderItems
//-----------------------------------------------------------------------


inline void BorderSet::getSomeBorderItems

  ( const IdxVector&  iitems,
    const IdxVector&  index,
    idx_t               iborder ) const

{
  borders_().getSomeGroupMembers ( iitems, index, iborder );
}


//-----------------------------------------------------------------------
//   toVector
//-----------------------------------------------------------------------


inline IdxVector BorderSet::toVector () const
{
  return borders_().toVector ();
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


inline BorderSet::Topology BorderSet::toMatrix () const
{
  return borders_().toMatrix ();
}


//-----------------------------------------------------------------------
//   getMembersOf
//-----------------------------------------------------------------------


inline IdxVector BorderSet::getMembersOf

  ( const IdxVector&  iborders ) const

{
  return borders_().getMembersOf ( iborders );
}


//-----------------------------------------------------------------------
//   getUniqueMembersOf
//-----------------------------------------------------------------------


inline IdxVector BorderSet::getUniqueMembersOf

  ( const IdxVector&  iborders ) const

{
  return borders_().getUniqueMembersOf ( iborders );
}


//-----------------------------------------------------------------------
//   borders_
//-----------------------------------------------------------------------


inline BorderSet::Data& BorderSet::borders_ () const
{
  return jem::staticCast<Data&> ( data_ );
}


JIVE_END_PACKAGE( mp )

#endif
