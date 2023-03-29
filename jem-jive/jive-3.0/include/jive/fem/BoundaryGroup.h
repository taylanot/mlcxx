
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

#ifndef JIVE_FEM_BOUNDARYGROUP_H
#define JIVE_FEM_BOUNDARYGROUP_H

#include <jive/util/ItemGroupSkin.h>
#include <jive/fem/import.h>
#include <jive/fem/BoundarySet.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class BoundaryGroup
//-----------------------------------------------------------------------


class BoundaryGroup : public ItemGroupSkin
{
 public:

  typedef BoundaryGroup     Self;
  typedef ItemGroupSkin     Super;

  typedef util::Topology    Topology;
  typedef util::TopoVector  TopoVector;


  inline                    BoundaryGroup   ()       noexcept;

  explicit inline           BoundaryGroup

    ( const Ref<Data>&        data )                 noexcept;

  explicit inline           BoundaryGroup

    ( Ref<Data>&&             data )                 noexcept;

  inline BoundarySet        getBoundaries   () const;

  inline void               checkBoundaries

    ( const String&           context,
      const Topology&         elemTopo )       const;

  inline void               checkBoundaries

    ( const String&           context,
      const IdxVector&        topoMap,
      const TopoVector&       elemTopos )      const;

  inline IdxVector          getNodeIndices

    ( const Topology&         elemTopo )       const;

  inline IdxVector          getNodeIndices

    ( const IdxVector&        topoMap,
      const TopoVector&       elemTopos )      const;

  static inline Self        find

    ( const String&           name,
      const BoundarySet&      bounds,
      const Properties&       globdat );

  static inline Self        get

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

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


BoundaryGroup               newBoundaryGroup

  ( const BoundarySet&        bounds );

BoundaryGroup               newBoundaryGroup

  ( const SliceAll&           all,
    const BoundarySet&        bounds );

BoundaryGroup               newBoundaryGroup

  ( const IdxVector&          ibounds,
    const BoundarySet&        bounds );





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline BoundaryGroup::BoundaryGroup () noexcept
{}


inline BoundaryGroup::BoundaryGroup

  ( const Ref<Data>&  data ) noexcept :

    Super ( data )

{}


inline BoundaryGroup::BoundaryGroup

  ( Ref<Data>&&  data ) noexcept :

    Super ( std::move( data ) )

{}


//-----------------------------------------------------------------------
//   getBoundaries
//-----------------------------------------------------------------------


inline BoundarySet BoundaryGroup::getBoundaries () const
{
  return BoundarySet (
    static_cast<BoundarySet::Data*> ( data_->getItems() )
  );
}


//-----------------------------------------------------------------------
//   checkBoundaries
//-----------------------------------------------------------------------


inline void BoundaryGroup::checkBoundaries

  ( const String&    context,
    const Topology&  elemTopo ) const

{
  getBoundaries().checkSomeBoundaries ( context,
                                        getIndices(), elemTopo );
}


inline void BoundaryGroup::checkBoundaries

  ( const String&      context,
    const IdxVector&   topoMap,
    const TopoVector&  elemTopos ) const

{
  getBoundaries().checkSomeBoundaries ( context, getIndices(),
                                        topoMap, elemTopos );
}


//-----------------------------------------------------------------------
//   getNodeIndices
//-----------------------------------------------------------------------


inline IdxVector BoundaryGroup::getNodeIndices

  ( const Topology&  elemTopo ) const

{
  return getBoundaries().getUniqueNodesOf ( getIndices(), elemTopo );
}


inline IdxVector BoundaryGroup::getNodeIndices

  ( const IdxVector&   topoMap,
    const TopoVector&  elemTopos ) const

{
  return getBoundaries().getUniqueNodesOf ( getIndices(),
                                            topoMap, elemTopos );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline BoundaryGroup  BoundaryGroup::find

  ( const String&       name,
    const BoundarySet&  bounds,
    const Properties&   globdat )

{
  return Self ( Data::find( name, bounds.getData(), globdat ) );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


inline BoundaryGroup  BoundaryGroup::get

  ( const String&       name,
    const BoundarySet&  bounds,
    const Properties&   globdat,
    const String&       context )

{
  return Self ( Data::get( name, bounds.getData(),
                           globdat, context ) );
}


//-----------------------------------------------------------------------
//   checkData
//-----------------------------------------------------------------------


inline void BoundaryGroup::checkData

  ( const Ref<Data>&  data,
    const String&     context )

{
  if ( ! isValidData( data ) )
  {
    dataError ( context, "boundary", data );
  }
}


JIVE_END_PACKAGE( fem )

#endif
