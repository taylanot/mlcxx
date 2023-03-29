
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

#ifndef JIVE_FEM_XNODESET_H
#define JIVE_FEM_XNODESET_H

#include <jive/util/StorageMode.h>
#include <jive/util/XPointSet.h>
#include <jive/fem/NodeSet.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class XNodeSet
//-----------------------------------------------------------------------


class XNodeSet : public NodeSet
{
 public:

  typedef XNodeSet          Self;
  typedef NodeSet           Super;
  typedef util::XPointSet   Data;


  inline                    XNodeSet    ()       noexcept;

  explicit inline           XNodeSet

    ( const Ref<Data>&        xnodes )           noexcept;

  explicit inline           XNodeSet

    ( Ref<Data>&&             xnodes )           noexcept;

  inline Data*              getData     () const noexcept;
  inline Self               clone       () const;
  inline void               clear       () const;

  inline void               reserve

    ( idx_t                   count )      const;

  inline void               trimToSize  () const;

  inline idx_t              addNode

    ( const Vector&           coords )     const;

  inline idx_t              addNode

    ( idx_t                   nodeID,
      const Vector&           coords )     const;

  inline void               eraseNodes

    ( const IdxVector&        inodes )     const;

  inline void               reorderNodes

    ( const Reordering&       reord )      const;

  inline void               setNodeCoords

    ( idx_t                   inode,
      const Vector&           coords )     const;

  inline void               setCoords

    ( const Matrix&           coords )     const;

  inline void               setSomeCoords

    ( const IdxVector&        inodes,
      const Matrix&           coords )     const;

  static inline Self        find

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


 private:

  inline Data&              xnodes_     () const;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline XNodeSet             toXNodeSet

  ( const NodeSet&            nodes );

XNodeSet                    newXNodeSet

  ( util::StorageMode         mode = util::DEFAULT_STORAGE );

XNodeSet                    newXNodeSet

  ( const String&             name,
    util::StorageMode         mode = util::DEFAULT_STORAGE );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class XNodeSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline XNodeSet::XNodeSet () noexcept
{}


inline XNodeSet::XNodeSet ( const Ref<Data>& xnodes ) noexcept :

  Super ( xnodes )

{}


inline XNodeSet::XNodeSet ( Ref<Data>&& xnodes ) noexcept :

  Super ( std::move( xnodes ) )

{}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


inline XNodeSet::Data* XNodeSet::getData () const noexcept
{
  return jem::staticCast<Data*> ( data_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


inline XNodeSet XNodeSet::clone () const
{
  return Self ( jem::checkedCast<Data>( data_->clone() ) );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


inline void XNodeSet::clear () const
{
  xnodes_().clear ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


inline void XNodeSet::reserve ( idx_t n ) const
{
  xnodes_().reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


inline void XNodeSet::trimToSize () const
{
  xnodes_().trimToSize ();
}


//-----------------------------------------------------------------------
//   addNode
//-----------------------------------------------------------------------


inline idx_t XNodeSet::addNode

  ( const Vector& coords ) const

{
  return xnodes_().addPoint ( coords );
}


inline idx_t XNodeSet::addNode

  ( idx_t          nodeID,
    const Vector&  coords ) const

{
  return xnodes_().addPoint ( nodeID, coords );
}


//-----------------------------------------------------------------------
//   eraseNodes
//-----------------------------------------------------------------------


inline void XNodeSet::eraseNodes ( const IdxVector& inodes ) const
{
  xnodes_().erasePoints ( inodes );
}


//-----------------------------------------------------------------------
//   reorderNodes
//-----------------------------------------------------------------------


inline void XNodeSet::reorderNodes ( const Reordering& reord ) const
{
  xnodes_().reorderPoints ( reord );
}


//-----------------------------------------------------------------------
//   setNodeCoords
//-----------------------------------------------------------------------


inline void XNodeSet::setNodeCoords

  ( idx_t          inode,
    const Vector&  coords ) const

{
  xnodes_().setPointCoords ( inode, coords );
}


//-----------------------------------------------------------------------
//   setCoords
//-----------------------------------------------------------------------


inline void XNodeSet::setCoords

  ( const Matrix&  coords ) const

{
  xnodes_().setCoords ( coords );
}


//-----------------------------------------------------------------------
//   setSomeCoords
//-----------------------------------------------------------------------


inline void XNodeSet::setSomeCoords

  ( const IdxVector&  inodes,
    const Matrix&     coords ) const

{
  xnodes_().setSomeCoords ( inodes, coords );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline XNodeSet XNodeSet::find ( const Properties& globdat )
{
  return toXNodeSet ( Super::find( globdat ) );
}


inline XNodeSet XNodeSet::find

  ( const String&      name,
    const Properties&  globdat )

{
  return toXNodeSet ( Super::find( name, globdat ) );
}


//-----------------------------------------------------------------------
//   xnodes_
//-----------------------------------------------------------------------


inline XNodeSet::Data& XNodeSet::xnodes_ () const
{
  return jem::staticCast<Data&> ( data_ );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   toXNodeSet
//-----------------------------------------------------------------------


inline XNodeSet toXNodeSet ( const NodeSet& nodes )
{
  return XNodeSet

    ( dynamic_cast<util::XPointSet*>( nodes.getData() ) );
}


JIVE_END_PACKAGE( fem )

#endif
