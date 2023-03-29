
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

#ifndef JIVE_MBODY_BODYSET_H
#define JIVE_MBODY_BODYSET_H

#include <jive/util/GroupSet.h>
#include <jive/fem/ElementSet.h>
#include <jive/mbody/import.h>


JIVE_BEGIN_PACKAGE( mbody )


using jem::util::Event;


//-----------------------------------------------------------------------
//   enum BodyKind
//-----------------------------------------------------------------------


enum BodyKind
{
  REGULAR_BODIES = 0,
  SURFACE_BODIES = 1
};


//-----------------------------------------------------------------------
//   class BodySet
//-----------------------------------------------------------------------


class BodySet : public ItemSetSkin
{
 public:

  typedef BodySet           Self;
  typedef ItemSetSkin       Super;

  typedef util::GroupSet    Data;
  typedef util::Topology    Topology;


  inline                    BodySet           ()       noexcept;

  explicit inline           BodySet

    ( const Ref<Data>&        bodies )                 noexcept;

  explicit inline           BodySet

    ( Ref<Data>&&             bodies )                 noexcept;

  inline Event
    < Data& >&              newTopoEvent      () const;

  inline idx_t              rank              () const;
  inline Data*              getData           () const noexcept;
  inline ElementSet         getElements       () const;

  inline String             getBodyPath

    ( idx_t                   bodyID )           const;

  inline idx_t              findBody

    ( idx_t                   bodyID )           const;

  inline idx_t              getBodyID

    ( idx_t                   ibody )            const;

  inline idx_t              maxBodyElemCount  () const;

  inline idx_t              maxBodyElemCountOf

    ( const IdxVector&        ibodies )          const;

  inline idx_t              getBodyElemCount

    ( idx_t                   ibody )            const;

  inline idx_t              getBodyElems

    ( const IdxVector&        ielems,
      idx_t                   ibody )            const;

  inline void               getSomeBodyElems

    ( const IdxVector&        ielems,
      const IdxVector&        index,
      idx_t                   ibody  )           const;

  inline IdxVector          toVector          () const;
  inline Topology           toMatrix          () const;

  inline IdxVector          getElemsOf

    ( const IdxVector&        ibodies )          const;

  IdxVector                 getNodesOf

    ( const IdxVector&        ibodies )          const;

  IdxVector                 getUniqueNodesOf

    ( const IdxVector&        ibodies )          const;

  void                      store

    ( BodyKind                kind,
      const Properties&       globdat )          const;

  void                      store

    ( const String&           name,
      const Properties&       globdat )          const;

  static Self               find

    ( BodyKind                kind,
      const Properties&       globdat );

  static Self               find

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

  static bool               isValidData

    ( const Ref<Data>&        data );

  static inline void        checkData

    ( const Ref<Data>&        data,
      const String&           context );


 protected:

  inline Data&              bodies_           () const;


 private:

  static const char*        CLASS_NAME_;

};





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class BodySet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline BodySet::BodySet () noexcept
{}


inline BodySet::BodySet ( const Ref<Data>&  bodies ) noexcept :

  Super ( bodies )

{}


inline BodySet::BodySet ( Ref<Data>&&  bodies ) noexcept :

  Super ( std::move( bodies ) )

{}


//-----------------------------------------------------------------------
//   newTopoEvent
//-----------------------------------------------------------------------


inline Event < util::GroupSet& >& BodySet::newTopoEvent () const
{
  return bodies_().newTopoEvent;
}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


inline idx_t BodySet::rank () const
{
  return getElements().getNodes().rank ();
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


inline BodySet::Data* BodySet::getData () const noexcept
{
  return jem::staticCast<Data*> ( data_ );
}


//-----------------------------------------------------------------------
//   getElements
//-----------------------------------------------------------------------


inline ElementSet BodySet::getElements () const
{
  return ElementSet (
    static_cast<ElementSet::Data*> ( bodies_().getGroupedItems() )
  );
}


//-----------------------------------------------------------------------
//   getBodyPath
//-----------------------------------------------------------------------


inline String BodySet::getBodyPath ( idx_t bodyID ) const
{
  return bodies_().getItemPath ( bodyID );
}


//-----------------------------------------------------------------------
//   findBody
//-----------------------------------------------------------------------


inline idx_t BodySet::findBody ( idx_t bodyID ) const
{
  return bodies_().findItem ( bodyID );
}


//-----------------------------------------------------------------------
//   getBodyID
//-----------------------------------------------------------------------


inline idx_t BodySet::getBodyID ( idx_t ibody  ) const
{
  return bodies_().getItemID ( ibody );
}


//-----------------------------------------------------------------------
//   maxBodyElemCount
//-----------------------------------------------------------------------


inline idx_t BodySet::maxBodyElemCount () const
{
  return bodies_().maxGroupSize ();
}


//-----------------------------------------------------------------------
//   maxBodyElemCountOf
//-----------------------------------------------------------------------


inline idx_t BodySet::maxBodyElemCountOf

  ( const IdxVector& ibodies ) const

{
  return bodies_().maxGroupSizeOf ( ibodies );
}


//-----------------------------------------------------------------------
//   getBodyElemCount
//-----------------------------------------------------------------------


inline idx_t BodySet::getBodyElemCount ( idx_t ibody ) const
{
  return bodies_().getGroupSize ( ibody );
}


//-----------------------------------------------------------------------
//   getBodyElems
//-----------------------------------------------------------------------


inline idx_t BodySet::getBodyElems

  ( const IdxVector&  inodes,
    idx_t             ibody ) const

{
  return bodies_().getGroupMembers ( inodes, ibody );
}


//-----------------------------------------------------------------------
//   getSomeBodyElems
//-----------------------------------------------------------------------


inline void BodySet::getSomeBodyElems

  ( const IdxVector&  inodes,
    const IdxVector&  index,
    idx_t             ibody  ) const

{
  bodies_().getSomeGroupMembers ( inodes, index, ibody );
}


//-----------------------------------------------------------------------
//   toVector
//-----------------------------------------------------------------------


inline IdxVector BodySet::toVector () const
{
  return bodies_().toVector ();
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


inline BodySet::Topology BodySet::toMatrix () const
{
  return bodies_().toMatrix ();
}


//-----------------------------------------------------------------------
//   getElemsOf
//-----------------------------------------------------------------------


inline IdxVector BodySet::getElemsOf

  ( const IdxVector&  ibodies ) const

{
  return bodies_().getMembersOf ( ibodies );
}


//-----------------------------------------------------------------------
//   checkData
//-----------------------------------------------------------------------


inline void BodySet::checkData

  ( const Ref<Data>&  data,
    const String&     context )

{
  if ( ! isValidData( data ) )
  {
    dataError ( context, "body", data );
  }
}


//-----------------------------------------------------------------------
//   bodies_
//-----------------------------------------------------------------------


inline BodySet::Data& BodySet::bodies_ () const
{
  return jem::staticCast<Data&> ( data_ );
}


JIVE_END_PACKAGE( mbody )

#endif
