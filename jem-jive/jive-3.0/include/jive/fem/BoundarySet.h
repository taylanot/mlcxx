
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

#ifndef JIVE_FEM_BOUNDARYSET_H
#define JIVE_FEM_BOUNDARYSET_H

#include <jive/util/MemberSet.h>
#include <jive/fem/ElementSet.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class BoundarySet
//-----------------------------------------------------------------------


class BoundarySet : public ItemSetSkin
{
 public:

  typedef BoundarySet       Self;
  typedef ItemSetSkin       Super;

  typedef util::MemberSet   Data;
  typedef util::Topology    Topology;
  typedef util::TopoVector  TopoVector;


  inline                    BoundarySet          ()      noexcept;

  explicit inline           BoundarySet

    ( const Ref<Data>&        bounds )                   noexcept;

  explicit inline           BoundarySet

    ( Ref<Data>&&             bounds )                   noexcept;

  inline Data*              getData             () const noexcept;
  inline ElementSet         getElements         () const;
  inline NodeSet            getNodes            () const;

  inline String             getBoundaryPath

    ( idx_t                   bndID )              const;

  inline idx_t              findBoundary

    ( idx_t                   bndID )              const;

  inline idx_t              getBoundaryID

    ( idx_t                   ibound )             const;

  inline void               getBoundary

    ( idx_t&                  ielem,
      idx_t&                  ilocal,
      idx_t                   ibound )             const;

  inline idx_t              getElemIndex

    ( idx_t                   ibound )             const;

  inline idx_t              getLocalIndex

    ( idx_t                   ibound )             const;

  void                      checkBoundaries

    ( const String&           context,
      const Topology&         elemTopo )           const;

  void                      checkBoundaries

    ( const String&           context,
      const IdxVector&        topoMap,
      const TopoVector&       elemTopos )          const;

  void                      checkSomeBoundaries

    ( const String&           context,
      const IdxVector&        ibounds,
      const Topology&         elemTopo )           const;

  void                      checkSomeBoundaries

    ( const String&           context,
      const IdxVector&        ibounds,
      const IdxVector&        topoMap,
      const TopoVector&       elemTopos )          const;

  IdxVector                 getNodesOf

    ( const IdxVector&        ibounds,
      const Topology&         elemTopo )           const;

  IdxVector                 getNodesOf

    ( const IdxVector&        ibounds,
      const IdxVector&        topoMap,
      const TopoVector&       elemTopos )          const;

  IdxVector                 getUniqueNodesOf

    ( const IdxVector&        ibounds,
      const Topology&         elemTopo )           const;

  IdxVector                 getUniqueNodesOf

    ( const IdxVector&        ibounds,
      const IdxVector&        topoMap,
      const TopoVector&       elemTopos )          const;

  inline IdxVector          getBoundsOf

    ( const IdxVector&        ielems )             const;

  static inline IdxVector   getBoundsOf

    ( const IdxVector&        ielems,
      const Topology&         beTopo );

  inline Topology           getBoundsPerElement () const;

  Topology                  getNodesPerBoundary

    ( const Topology&         elemTopo )           const;

  Topology                  getNodesPerBoundary

    ( const IdxVector&        topoMap,
      const TopoVector&       elemTopos )          const;

  void                      findBuddies

    ( const IdxVector&        buddies,
      const Topology&         elemTopo )           const;

  void                      findBuddies

    ( const IdxVector&        buddies,
      const IdxVector&        topoMap,
      const TopoVector&       elemTopos )          const;

  void                      store

    ( const Properties&       globdat )            const;

  void                      store

    ( const String&           name,
      const Properties&       globdat )            const;

  static Self               find

    ( const Properties&       globdat );

  static Self               find

    ( const String&           name,
      const Properties&       globdat );

  static Self               get

    ( const Properties&       globdat,
      const String&           context );

  static Self               get

    ( const String&           name,
      const Properties&       globdat,
      const String&           context );

  static bool               isValidData

    ( const Ref<Data>&        data );

  static inline void        checkData

    ( const Ref<Data>&        data,
      const String&           context );


 protected:

  inline Data&              bounds_             () const;


 private:

  static const char*        CLASS_NAME_;

};







//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class BoundarySet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline BoundarySet::BoundarySet () noexcept
{}


inline BoundarySet::BoundarySet

  ( const Ref<Data>&  bounds ) noexcept :

    Super ( bounds )

{}


inline BoundarySet::BoundarySet

  ( Ref<Data>&&  bounds ) noexcept :

    Super ( std::move( bounds ) )

{}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


inline BoundarySet::Data* BoundarySet::getData () const noexcept
{
  return jem::staticCast<Data*> ( data_ );
}


//-----------------------------------------------------------------------
//   getElements
//-----------------------------------------------------------------------


inline ElementSet BoundarySet::getElements () const
{
  return ElementSet (
    static_cast<ElementSet::Data*> ( bounds_().getCompoundItems() )
  );
}


//-----------------------------------------------------------------------
//   getNodes
//-----------------------------------------------------------------------


inline NodeSet BoundarySet::getNodes () const
{
  ElementSet::Data*  elems =

    static_cast<ElementSet::Data*> ( bounds_().getCompoundItems() );

  return NodeSet (
    static_cast<NodeSet::Data*> ( elems->getGroupedItems() )
  );
}


//-----------------------------------------------------------------------
//   getBoundaryPath
//-----------------------------------------------------------------------


inline String BoundarySet::getBoundaryPath ( idx_t bndID ) const
{
  return bounds_().getItemPath ( bndID );
}


//-----------------------------------------------------------------------
//   findBoundary
//-----------------------------------------------------------------------


inline idx_t BoundarySet::findBoundary ( idx_t bndID ) const
{
  return bounds_().findItem ( bndID );
}


//-----------------------------------------------------------------------
//   getBoundaryID
//-----------------------------------------------------------------------


inline idx_t BoundarySet::getBoundaryID ( idx_t inode  ) const
{
  return bounds_().getItemID ( inode );
}


//-----------------------------------------------------------------------
//   getBoundary
//-----------------------------------------------------------------------


inline void BoundarySet::getBoundary

  ( idx_t&  ielem,
    idx_t&  ilocal,
    idx_t   ibound ) const

{
  bounds_().getMember ( ielem, ilocal, ibound );
}


//-----------------------------------------------------------------------
//   getElemIndex
//-----------------------------------------------------------------------


inline idx_t BoundarySet::getElemIndex ( idx_t ibound ) const
{
  idx_t  ielem;
  idx_t  ilocal;

  bounds_().getMember ( ielem, ilocal, ibound );

  return ielem;
}


//-----------------------------------------------------------------------
//   getLocalIndex
//-----------------------------------------------------------------------


inline idx_t BoundarySet::getLocalIndex ( idx_t ibound ) const
{
  idx_t  ielem;
  idx_t  ilocal;

  bounds_().getMember ( ielem, ilocal, ibound );

  return ilocal;
}


//-----------------------------------------------------------------------
//   getBoundsOf
//-----------------------------------------------------------------------


inline IdxVector BoundarySet::getBoundsOf

  ( const IdxVector&  ielems ) const

{
  return bounds_().getMembersOf ( ielems );
}


inline IdxVector BoundarySet::getBoundsOf

  ( const IdxVector&  ielems,
    const Topology&   ebTopo )

{
  return Data::getMembersOf ( ielems, ebTopo );
}


//-----------------------------------------------------------------------
//   getBoundsPerElement
//-----------------------------------------------------------------------


inline BoundarySet::Topology

  BoundarySet::getBoundsPerElement () const

{
  return bounds_().getMembersPerItem ();
}


//-----------------------------------------------------------------------
//   checkData
//-----------------------------------------------------------------------


inline void BoundarySet::checkData

  ( const Ref<Data>&  data,
    const String&     context )

{
  if ( ! isValidData( data ) )
  {
    dataError ( context, "boundary", data );
  }
}


//-----------------------------------------------------------------------
//   bounds
//-----------------------------------------------------------------------


inline BoundarySet::Data& BoundarySet::bounds_ () const
{
  return jem::staticCast<Data&> ( data_ );
}


JIVE_END_PACKAGE( fem )

#endif
