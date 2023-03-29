
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

#ifndef JIVE_FEM_XBOUNDARYSET_H
#define JIVE_FEM_XBOUNDARYSET_H

#include <jive/util/StorageMode.h>
#include <jive/util/XMemberSet.h>
#include <jive/fem/BoundarySet.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class XBoundarySet
//-----------------------------------------------------------------------


class XBoundarySet : public BoundarySet
{
 public:

  typedef XBoundarySet      Self;
  typedef BoundarySet       Super;
  typedef util::XMemberSet  Data;


  inline                    XBoundarySet  ()       noexcept;

  explicit inline           XBoundarySet

    ( const Ref<Data>&        bounds )             noexcept;

  explicit inline           XBoundarySet

    ( Ref<Data>&&             bounds )             noexcept;

  inline Data*              getData       () const noexcept;
  inline Self               clone         () const;
  inline void               clear         () const;

  inline void               reserve

    ( idx_t                   count )        const;

  inline void               trimToSize    () const;

  inline idx_t              addBoundary

    ( idx_t                   ielem,
      idx_t                   ilocal )       const;

  inline idx_t              addBoundary

    ( idx_t                   bndID,
      idx_t                   ielem,
      idx_t                   ilocal )       const;

  idx_t                     addBoundaries

    ( const Topology&         elemTopo  )    const;

  idx_t                     addBoundaries

    ( const IdxVector&        topoMap,
      const TopoVector&       elemTopos )    const;

  idx_t                     addBoundariesOf

    ( const IdxVector&        ielems,
      const Topology&         elemTopo  )    const;

  idx_t                     addBoundariesOf

    ( const IdxVector&        ielems,
      const IdxVector&        topoMap,
      const TopoVector&       elemTopos )    const;

  inline void               eraseBoundaries

    ( const IdxVector&        ibounds )      const;

  inline void               reorderBoundaries

    ( const Reordering&       reord )        const;

   static inline Self       find

    ( const Properties&       globdat );

  static inline Self        find

    ( const String&           name,
      const Properties&       globdat );

  static Self               get

    ( const Properties&       globdat,
      const String&           context );

  static Self               get

    ( const String&           name,
      const Properties&       globdat,
      const String&           context );


 protected:

  inline Data&              xbounds_      () const;


 private:

  static const char*        CLASS_NAME_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline XBoundarySet         toXBoundarySet

  ( const BoundarySet&        bounds );

XBoundarySet                newXBoundarySet

  ( const ElementSet&         elems,
    util::StorageMode         mode = util::DEFAULT_STORAGE );

XBoundarySet                newXBoundarySet

  ( const String&             name,
    const ElementSet&         elems,
    util::StorageMode         mode = util::DEFAULT_STORAGE );






//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class XBoundarySet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline XBoundarySet::XBoundarySet () noexcept
{}


inline XBoundarySet::XBoundarySet

  ( const Ref<Data>&  bounds ) noexcept :

    Super ( bounds )

{}


inline XBoundarySet::XBoundarySet

  ( Ref<Data>&&  bounds ) noexcept :

    Super ( std::move( bounds ) )

{}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


inline XBoundarySet::Data* XBoundarySet::getData () const noexcept
{
  return jem::staticCast<Data*> ( data_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


inline XBoundarySet XBoundarySet::clone () const
{
  return Self ( jem::checkedCast<Data>( data_->clone() ) );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


inline void XBoundarySet::clear () const
{
  xbounds_().clear ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


inline void XBoundarySet::reserve ( idx_t n ) const
{
  xbounds_().reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


inline void XBoundarySet::trimToSize () const
{
  xbounds_().trimToSize ();
}


//-----------------------------------------------------------------------
//   addBoundary
//-----------------------------------------------------------------------


inline idx_t XBoundarySet::addBoundary

  ( idx_t  ielem,
    idx_t  ilocal ) const

{
  return xbounds_().addMember ( ielem, ilocal );
}


inline idx_t XBoundarySet::addBoundary

  ( idx_t  bndID,
    idx_t  ielem,
    idx_t  ilocal ) const

{
  return xbounds_().addMember ( bndID, ielem, ilocal );
}


//-----------------------------------------------------------------------
//   eraseBoundaries
//-----------------------------------------------------------------------


inline void XBoundarySet::eraseBoundaries

  ( const IdxVector& ibounds ) const

{
  xbounds_().eraseMembers ( ibounds );
}


//-----------------------------------------------------------------------
//   reorderBoundaries
//-----------------------------------------------------------------------


inline void XBoundarySet::reorderBoundaries

  ( const Reordering& reord ) const

{
  xbounds_().reorderMembers ( reord );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline XBoundarySet XBoundarySet::find ( const Properties& globdat )
{
  return toXBoundarySet ( Super::find( globdat ) );
}


inline XBoundarySet XBoundarySet::find

  ( const String&      name,
    const Properties&  globdat )

{
  return toXBoundarySet ( Super::find( name, globdat ) );
}


//-----------------------------------------------------------------------
//   xbounds_
//-----------------------------------------------------------------------


inline XBoundarySet::Data& XBoundarySet::xbounds_ () const
{
  return jem::staticCast<Data&> ( data_ );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   toXBoundarySet
//-----------------------------------------------------------------------


inline XBoundarySet toXBoundarySet ( const BoundarySet& bounds )
{
  return XBoundarySet (
    dynamic_cast<util::XMemberSet*> ( bounds.getData() )
  );
}


JIVE_END_PACKAGE( fem )

#endif
