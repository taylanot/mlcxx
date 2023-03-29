
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

#ifndef JIVE_MBODY_XBODYSET_H
#define JIVE_MBODY_XBODYSET_H

#include <jive/util/StorageMode.h>
#include <jive/util/XGroupSet.h>
#include <jive/mbody/BodySet.h>


JIVE_BEGIN_PACKAGE( mbody )


//-----------------------------------------------------------------------
//   class XBodySet
//-----------------------------------------------------------------------


class XBodySet : public BodySet
{
 public:

  typedef XBodySet          Self;
  typedef BodySet           Super;

  typedef util::XGroupSet   Data;


  inline                    XBodySet ()          noexcept;

  explicit inline           XBodySet

    ( const Ref<Data>&        bodies )           noexcept;

  explicit inline           XBodySet

    ( Ref<Data>&&             bodies )           noexcept;

  inline Data*              getData     () const noexcept;
  inline Self               clone       () const;
  inline void               clear       () const;

  inline void               reserve

    ( idx_t                   count )      const;

  inline void               trimToSize  () const;

  inline idx_t              addBody

    ( const IdxVector&        ielems )     const;

  inline idx_t                addBody

    ( idx_t                   bodyID,
      const IdxVector&        ielems )     const;

  inline void               eraseBodies

    ( const IdxVector&        ibodies )    const;

  inline void               reorderBodies

    ( const Reordering&       reord )      const;

  inline void               setBodyElems

    ( idx_t                   ibody,
      const IdxVector&        ielems )     const;

  static inline Self        find

    ( BodyKind                kind,
      const Properties&       globdat );

  static inline Self        find

    ( const String&           name,
      const Properties&       globdat );

  static Self               get

    ( BodyKind                kind,
      const Properties&       globdat,
      const String&           context );

  static Self               get

    ( const String&           name,
      const Properties&       globdat,
      const String&           context );


 private:

  inline Data&              xbodies_    () const;


 private:

  static const char*        CLASS_NAME_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline XBodySet             toXBodySet

  ( const BodySet&            bodies );

XBodySet                    newXBodySet

  ( BodyKind                  kind,
    const ElementSet&         elems,
    util::StorageMode         mode = util::DEFAULT_STORAGE );

XBodySet                    newXBodySet

  ( const String&             name,
    const ElementSet&         elems,
    util::StorageMode         mode = util::DEFAULT_STORAGE );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class XBodySet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline XBodySet::XBodySet () noexcept
{}


inline XBodySet::XBodySet ( const Ref<Data>& bodies ) noexcept :

    Super ( bodies )

{}


inline XBodySet::XBodySet ( Ref<Data>&& bodies ) noexcept :

    Super ( std::move( bodies ) )

{}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


inline XBodySet::Data* XBodySet::getData () const noexcept
{
  return jem::staticCast<Data*> ( data_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


inline XBodySet XBodySet::clone () const
{
  return Self ( jem::checkedCast<Data>( data_->clone() ) );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


inline void XBodySet::clear () const
{
  xbodies_().clear ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


inline void XBodySet::reserve ( idx_t n ) const
{
  xbodies_().reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


inline void XBodySet::trimToSize () const
{
  xbodies_().trimToSize ();
}


//-----------------------------------------------------------------------
//   addBody
//-----------------------------------------------------------------------


inline idx_t XBodySet::addBody ( const IdxVector& ielems ) const
{
  return xbodies_().addGroup ( ielems );
}


inline idx_t XBodySet::addBody

  ( idx_t             bodyID,
    const IdxVector&  ielems ) const

{
  return xbodies_().addGroup ( bodyID, ielems );
}


//-----------------------------------------------------------------------
//   eraseBodies
//-----------------------------------------------------------------------


inline void XBodySet::eraseBodies

  ( const IdxVector& ibodies ) const

{
  xbodies_().eraseGroups ( ibodies );
}


//-----------------------------------------------------------------------
//   reorderBodies
//-----------------------------------------------------------------------


inline void XBodySet::reorderBodies

  ( const Reordering& reord ) const

{
  xbodies_().reorderGroups ( reord );
}


//-----------------------------------------------------------------------
//   setBodyElems
//-----------------------------------------------------------------------


inline void XBodySet::setBodyElems

  ( idx_t             ibody,
    const IdxVector&  ielems ) const

{
  xbodies_().setGroupMembers ( ibody, ielems );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline XBodySet XBodySet::find

  ( BodyKind           kind,
    const Properties&  globdat )

{
  return toXBodySet ( Super::find( kind, globdat ) );
}


inline XBodySet XBodySet::find

  ( const String&      name,
    const Properties&  globdat )

{
  return toXBodySet ( Super::find( name, globdat ) );
}


//-----------------------------------------------------------------------
//   xbodies_
//-----------------------------------------------------------------------


inline XBodySet::Data& XBodySet::xbodies_ () const
{
  return jem::staticCast<Data&> ( data_ );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   toXBodySet
//-----------------------------------------------------------------------


inline XBodySet toXBodySet ( const BodySet& bodies )
{
  return XBodySet

    ( dynamic_cast<util::XGroupSet*>( bodies.getData() ) );
}


JIVE_END_PACKAGE( mbody )

#endif
