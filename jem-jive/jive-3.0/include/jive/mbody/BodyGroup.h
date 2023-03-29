
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

#ifndef JIVE_MBODY_BODYGROUP_H
#define JIVE_MBODY_BODYGROUP_H

#include <jive/util/ItemGroupSkin.h>
#include <jive/mbody/BodySet.h>


JIVE_BEGIN_PACKAGE( mbody )


//-----------------------------------------------------------------------
//   class BodyGroup
//-----------------------------------------------------------------------


class BodyGroup : public ItemGroupSkin
{
 public:

  typedef BodyGroup         Self;
  typedef ItemGroupSkin     Super;


  inline                    BodyGroup         ()       noexcept;

  inline                    BodyGroup

    ( const Ref<Data>&        data )                   noexcept;

  inline                    BodyGroup

    ( Ref<Data>&&             data )                   noexcept;

  inline BodySet            getBodies         () const;
  inline IdxVector          getElemIndices    () const;
  inline IdxVector          getNodeIndices    () const;

  static inline Self        find

    ( const String&           name,
      const BodySet&          bodies,
      const Properties&       globdat );

  static inline Self        get

    ( const String&           name,
      const BodySet&          bodies,
      const Properties&       globdat,
      const String&           context );

  static Self               get

    ( BodyKind                kind,
      const Properties&       conf,
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


BodyGroup                   newBodyGroup

  ( const BodySet&            bodies );

BodyGroup                   newBodyGroup

  ( const SliceAll&           all,
    const BodySet&            bodies );

BodyGroup                   newBodyGroup

  ( const IdxVector&          ibodies,
    const BodySet&            bodies );




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline BodyGroup::BodyGroup () noexcept
{}


inline BodyGroup::BodyGroup ( const Ref<Data>& data ) noexcept :

  Super ( data )

{}


inline BodyGroup::BodyGroup ( Ref<Data>&& data ) noexcept :

  Super ( std::move( data ) )

{}


//-----------------------------------------------------------------------
//   getBodies
//-----------------------------------------------------------------------


inline BodySet BodyGroup::getBodies () const
{
  return BodySet

    ( static_cast<BodySet::Data*>( data_->getItems() ) );
}


//-----------------------------------------------------------------------
//   getElemIndices
//-----------------------------------------------------------------------


inline IdxVector BodyGroup::getElemIndices () const
{
  return getBodies().getElemsOf ( getIndices() );
}


//-----------------------------------------------------------------------
//   getNodeIndices
//-----------------------------------------------------------------------


inline IdxVector BodyGroup::getNodeIndices () const
{
  return getBodies().getUniqueNodesOf ( getIndices() );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline BodyGroup BodyGroup::find

  ( const String&      name,
    const BodySet&     bodies,
    const Properties&  globdat )

{
  return Self ( Data::find( name, bodies.getData(), globdat ) );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


inline BodyGroup BodyGroup::get

  ( const String&      name,
    const BodySet&     bodies,
    const Properties&  globdat,
    const String&      context )

{
  return Self ( Data::get( name, bodies.getData(),
                           globdat, context ) );
}


//-----------------------------------------------------------------------
//   checkData
//-----------------------------------------------------------------------


inline void BodyGroup::checkData

  ( const Ref<Data>&  data,
    const String&     context )

{
  if ( ! isValidData( data ) )
  {
    dataError ( context, "body", data );
  }
}


JIVE_END_PACKAGE( mbody )

#endif
