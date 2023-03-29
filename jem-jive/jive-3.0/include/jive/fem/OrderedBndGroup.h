
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

#ifndef JIVE_FEM_ORDEREDBNDGROUP_H
#define JIVE_FEM_ORDEREDBNDGROUP_H

#include <jive/util/ColoredItemGroup.h>
#include <jive/fem/BoundaryGroup.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class OrderedBndGroup
//-----------------------------------------------------------------------


class OrderedBndGroup : public BoundaryGroup
{
 public:

  typedef OrderedBndGroup   Self;
  typedef BoundaryGroup     Super;

  typedef util::
    ColoredItemGroup        Data;


  inline                    OrderedBndGroup   ()       noexcept;

  explicit inline           OrderedBndGroup

    ( const Ref<Data>&        data )                   noexcept;

  explicit inline           OrderedBndGroup

    ( Ref<Data>&&             data )                   noexcept;

  inline Data*              getData           () const noexcept;
  inline idx_t              listCount         () const;

  inline IdxVector          getBoundaryList

    ( idx_t                   ilist )            const;

  inline idx_t              getListOffset

    ( idx_t                   ilist )            const;

  inline idx_t              getLocalIndex

    ( idx_t                   ilist )            const;

  static IdxVector          getColors

    ( const IdxVector&        ibounds,
      const BoundarySet&      bounds );

  void                      store

    ( const String&           name,
      const Properties&       globdat )          const;

  static Self               find

    ( const String&           name,
      const BoundarySet&      bounds,
      const Properties&       globdat );

  static Self               find

    ( const String&           name,
      const String&           bounds,
      const Properties&       globdat );

  static Self               get

    ( const String&           name,
      const BoundarySet&      bounds,
      const Properties&       globdat,
      const String&           context );

  static Self               get

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat,
      const String&           context );

  static bool               isValidData

    ( const Ref<Data>&        data );

  static inline void        checkData

    ( const Ref<Data>&        data,
      const String&           context );


 protected:

  inline Data&              cgroup_           () const;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


OrderedBndGroup             newOrderedBndGroup

  ( const BoundarySet&        bounds );

OrderedBndGroup             newOrderedBndGroup

  ( const IdxVector&          ibounds,
    const BoundarySet&        bounds );





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline OrderedBndGroup::OrderedBndGroup () noexcept
{}


inline OrderedBndGroup::OrderedBndGroup

  ( const Ref<Data>&  data ) noexcept :

    Super ( data )

{}


inline OrderedBndGroup::OrderedBndGroup

  ( Ref<Data>&&  data ) noexcept :

    Super ( std::move( data ) )

{}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


inline OrderedBndGroup::Data*
  OrderedBndGroup::getData () const noexcept
{
  return jem::staticCast<Data*> ( data_ );
}


//-----------------------------------------------------------------------
//   listCount
//-----------------------------------------------------------------------


inline idx_t OrderedBndGroup::listCount () const
{
  return cgroup_().listCount ();
}


//-----------------------------------------------------------------------
//   getBoundaryList
//-----------------------------------------------------------------------


inline IdxVector
  OrderedBndGroup::getBoundaryList ( idx_t ilist ) const
{
  return cgroup_().getList ( ilist );
}


//-----------------------------------------------------------------------
//   getListOffset
//-----------------------------------------------------------------------


inline idx_t OrderedBndGroup::getListOffset ( idx_t ilist ) const
{
  return cgroup_().getListOffset ( ilist );
}


//-----------------------------------------------------------------------
//   getLocalIndex
//-----------------------------------------------------------------------


inline idx_t OrderedBndGroup::getLocalIndex ( idx_t ilist ) const
{
  return cgroup_().getListColor ( ilist );
}


//-----------------------------------------------------------------------
//   checkData
//-----------------------------------------------------------------------


inline void OrderedBndGroup::checkData

  ( const Ref<Data>&  data,
    const String&     context )

{
  if ( ! isValidData( data ) )
  {
    dataError ( context, "boundary", data );
  }
}


//-----------------------------------------------------------------------
//   cgroup_
//-----------------------------------------------------------------------


inline OrderedBndGroup::Data& OrderedBndGroup::cgroup_ () const
{
  return jem::staticCast<Data&> ( data_ );
}


JIVE_END_PACKAGE( fem )

#endif
