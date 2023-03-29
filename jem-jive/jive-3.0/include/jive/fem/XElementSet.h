
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

#ifndef JIVE_FEM_XELEMENTSET_H
#define JIVE_FEM_XELEMENTSET_H

#include <jive/util/StorageMode.h>
#include <jive/util/XGroupSet.h>
#include <jive/fem/ElementSet.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class XElementSet
//-----------------------------------------------------------------------


class XElementSet : public ElementSet
{
 public:

  typedef XElementSet       Self;
  typedef ElementSet        Super;
  typedef util::XGroupSet   Data;


  inline                    XElementSet ()       noexcept;

  explicit inline           XElementSet

    ( const Ref<Data>&        elems )            noexcept;

  explicit inline           XElementSet

    ( Ref<Data>&&             elems )            noexcept;

  inline Data*              getData     () const noexcept;
  inline Self               clone       () const;
  inline void               clear       () const;

  inline void               reserve

    ( idx_t                   count )      const;

  inline void               trimToSize  () const;

  inline idx_t              addElement

    ( const IdxVector&        inodes )     const;

  inline idx_t              addElement

    ( idx_t                   elemID,
      const IdxVector&        inodes )     const;

  inline void               eraseElements

    ( const IdxVector&        ielems )     const;

  inline void               reorderElements

    ( const Reordering&       reord )      const;

  inline void               setElemNodes

    ( idx_t                   ielem,
      const IdxVector&        inodes )     const;

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

  inline Data&              xelems_     () const;


 private:

  static const char*        CLASS_NAME_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline XElementSet          toXElementSet

  ( const ElementSet&         elems );

XElementSet                 newXElementSet

  ( const NodeSet&            nodes,
    util::StorageMode         mode = util::DEFAULT_STORAGE );

XElementSet                 newXElementSet

  ( const String&             name,
    const NodeSet&            nodes,
    util::StorageMode         mode = util::DEFAULT_STORAGE );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class XElementSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline XElementSet::XElementSet () noexcept
{}


inline XElementSet::XElementSet

  ( const Ref<Data>&  elems ) noexcept :

    Super ( elems )

{}


inline XElementSet::XElementSet

  ( Ref<Data>&&  elems ) noexcept :

    Super ( std::move( elems ) )

{}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


inline XElementSet::Data* XElementSet::getData () const noexcept
{
  return jem::staticCast<Data*> ( data_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


inline XElementSet XElementSet::clone () const
{
  return Self ( jem::checkedCast<Data>( data_->clone() ) );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


inline void XElementSet::clear () const
{
  xelems_().clear ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


inline void XElementSet::reserve ( idx_t n ) const
{
  xelems_().reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


inline void XElementSet::trimToSize () const
{
  xelems_().trimToSize ();
}


//-----------------------------------------------------------------------
//   addElement
//-----------------------------------------------------------------------


inline idx_t XElementSet::addElement

  ( const IdxVector&  inodes ) const

{
  return xelems_().addGroup ( inodes );
}


inline idx_t XElementSet::addElement

  ( idx_t             elemID,
    const IdxVector&  inodes ) const

{
  return xelems_().addGroup ( elemID, inodes );
}


//-----------------------------------------------------------------------
//   eraseElements
//-----------------------------------------------------------------------


inline void XElementSet::eraseElements

  ( const IdxVector& ielems ) const

{
  xelems_().eraseGroups ( ielems );
}


//-----------------------------------------------------------------------
//   reorderElements
//-----------------------------------------------------------------------


inline void XElementSet::reorderElements

  ( const Reordering&  reord ) const

{
  xelems_().reorderGroups ( reord );
}


//-----------------------------------------------------------------------
//   setElemNodes
//-----------------------------------------------------------------------


inline void XElementSet::setElemNodes

  ( idx_t             ielem,
    const IdxVector&  inodes ) const

{
  xelems_().setGroupMembers ( ielem, inodes );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline XElementSet XElementSet::find ( const Properties& globdat )
{
  return toXElementSet ( Super::find( globdat ) );
}


inline XElementSet XElementSet::find

  ( const String&      name,
    const Properties&  globdat )

{
  return toXElementSet ( Super::find( name, globdat ) );
}


//-----------------------------------------------------------------------
//   xelems_
//-----------------------------------------------------------------------


inline XElementSet::Data& XElementSet::xelems_ () const
{
  return jem::staticCast<Data&> ( data_ );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   toXElementSet
//-----------------------------------------------------------------------


inline XElementSet toXElementSet ( const ElementSet& elems )
{
  return XElementSet (
    dynamic_cast<util::XGroupSet*> ( elems.getData() )
  );
}


JIVE_END_PACKAGE( fem )

#endif
