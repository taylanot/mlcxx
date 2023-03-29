
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

#ifndef JIVE_FEM_ELEMENTGROUP_H
#define JIVE_FEM_ELEMENTGROUP_H

#include <jive/util/ItemGroupSkin.h>
#include <jive/fem/ElementSet.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class ElementGroup
//-----------------------------------------------------------------------


class ElementGroup : public ItemGroupSkin
{
 public:

  typedef ElementGroup      Self;
  typedef ItemGroupSkin     Super;


  inline                    ElementGroup      ()       noexcept;

  explicit inline           ElementGroup

    ( const Ref<Data>&        data )                   noexcept;

  explicit inline           ElementGroup

    ( Ref<Data>&&             data )                   noexcept;

  inline ElementSet         getElements       () const;
  inline IdxVector          getNodeIndices    () const;

  inline void               checkElements

    ( const String&           context,
      idx_t                   nodeCount )        const;

  inline void               getElemBoxes

    ( const Matrix&            boxes )           const;

  static inline Self        find

    ( const String&           name,
      const ElementSet&       elems,
      const Properties&       globdat );

  static inline Self        get

    ( const String&           name,
      const ElementSet&       elems,
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


ElementGroup                newElementGroup

  ( const ElementSet&         elems );

ElementGroup                newElementGroup

  ( const SliceAll&           all,
    const ElementSet&         elems );

ElementGroup                newElementGroup

  ( const IdxVector&          ielems,
    const ElementSet&         elems );





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline ElementGroup::ElementGroup () noexcept
{}


inline ElementGroup::ElementGroup

  ( const Ref<Data>&  data ) noexcept :

    Super ( data )

{}


inline ElementGroup::ElementGroup

  ( Ref<Data>&&  data ) noexcept :

    Super ( std::move( data ) )

{}


//-----------------------------------------------------------------------
//   getElements
//-----------------------------------------------------------------------


inline ElementSet ElementGroup::getElements () const
{
  return ElementSet (
    static_cast<ElementSet::Data*> ( data_->getItems() )
  );
}


//-----------------------------------------------------------------------
//   getNodeIndices
//-----------------------------------------------------------------------


inline IdxVector ElementGroup::getNodeIndices () const
{
  return getElements().getUniqueNodesOf ( getIndices() );
}


//-----------------------------------------------------------------------
//   checkElements
//-----------------------------------------------------------------------


inline void ElementGroup::checkElements

  ( const String&  context,
    idx_t            nodeCount ) const

{
  getElements().checkSomeElements ( context,
                                    getIndices(), nodeCount );
}


//-----------------------------------------------------------------------
//   getElemBoxes
//-----------------------------------------------------------------------


inline void ElementGroup::getElemBoxes ( const Matrix& boxes ) const
{
  getElements().getSomeElemBoxes ( boxes, getIndices() );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline ElementGroup ElementGroup::find

  ( const String&       name,
    const ElementSet&   elems,
    const Properties&   globdat )

{
  return Self ( Data::find( name, elems.getData(), globdat ) );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


inline ElementGroup ElementGroup::get

  ( const String&      name,
    const ElementSet&  elems,
    const Properties&  globdat,
    const String&      context )

{
  return Self ( Data::get( name, elems.getData(),
                           globdat, context ) );
}


//-----------------------------------------------------------------------
//   checkData
//-----------------------------------------------------------------------


inline void ElementGroup::checkData

  ( const Ref<Data>&  data,
    const String&     context )

{
  if ( ! isValidData( data ) )
  {
    dataError ( context, "element", data );
  }
}


JIVE_END_PACKAGE( fem )

#endif
