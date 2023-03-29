
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

#ifndef JIVE_UTIL_DOFSPACE_H
#define JIVE_UTIL_DOFSPACE_H

#include <jem/base/Object.h>
#include <jive/Array.h>
#include <jive/SparseMatrix.h>
#include <jive/util/import.h>
#include <jive/util/Reordering.h>
#include <jive/util/EventSource.h>


JIVE_BEGIN_PACKAGE( util )


class ItemSet;


//-----------------------------------------------------------------------
//   class DofSpace
//-----------------------------------------------------------------------


class DofSpace : public Object,
                 public EventSource
{
 public:

  JEM_DECLARE_CLASS       ( DofSpace, Object );

  Event
    < idx_t, Self& >        newSizeEvent;
  Event
    < const Reordering&,
      Self& >               newOrderEvent;


  virtual String            toString      () const override;
  idx_t                     itemCount     () const;
  virtual idx_t             typeCount     () const = 0;
  virtual idx_t             dofCount      () const = 0;
  inline  bool              isDense       () const;

  virtual idx_t             findType

    ( const String&           name )         const;

  inline  idx_t             getTypeIndex

    ( const String&           name )         const;

  virtual String            getTypeName

    ( idx_t                   itype )        const = 0;

  virtual StringVector      getTypeNames  () const;

  virtual idx_t             findDofIndex

    ( idx_t                   iitem,
      idx_t                   itype )        const = 0;

  inline idx_t              getDofIndex

    ( idx_t                   iitem,
      idx_t                   itype )        const;

  virtual idx_t             findDofIndices

    ( const IdxVector&        idofs,
      const IdxVector&        iitems,
      const IdxVector&        itypes )       const;

  virtual idx_t             collectDofIndices

    ( const IdxVector&        idofs,
      const IdxVector&        iitems,
      const IdxVector&        itypes )       const;

  inline void               getDofIndices

    ( const IdxVector&        idofs,
      const IdxVector&        iitems,
      const IdxVector&        itypes )       const;

  void                      getDofIndices

    ( const IdxVector&        idofs,
      idx_t                   iitem,
      const IdxVector&        itypes )       const;

  void                      getDofIndices

    ( const IdxVector&        idofs,
      const IdxVector&        iitems,
      idx_t                   itype )        const;

  virtual idx_t             getDofsForItem

    ( const IdxVector&        idofs,
      const IdxVector&        itypes,
      idx_t                   iitem )        const;

  virtual idx_t             getDofsForType

    ( const IdxVector&        idofs,
      const IdxVector&        iitems,
      idx_t                   itype )        const;

  virtual void              decodeDofIndex

    ( idx_t&                  iitem,
      idx_t&                  itype,
      idx_t                   idof   )       const = 0;

  virtual void              decodeDofIndices

    ( const IdxVector&        iitems,
      const IdxVector&        itypes,
      const IdxVector&        idofs  )       const;

  virtual SparseIdxMatrix   toMatrix      () const = 0;
  virtual ItemSet*          getItems      () const = 0;

  String                    getDofName

    ( idx_t                   idof )         const;

  String                    getDofName

    ( idx_t                   iitem,
      idx_t                   itype )        const;

  String                    getDofPath

    ( idx_t                   idof )         const;

  String                    getDofPath

    ( idx_t                   iitem,
      idx_t                   itype )        const;

  String                    getContext    () const;

  void                      printTo

    ( TextOutput&             out )          const;

  void                      store

    ( const Properties&       globdat )      const;

  static Ref<DofSpace>      find

    ( const Properties&       globdat );

  static Ref<DofSpace>      find

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Ref<DofSpace>      get

    ( const Properties&       globdat,
      const String&           context );

  static Ref<DofSpace>      get

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat,
      const String&           context );


 protected:

  virtual                  ~DofSpace      ();


 private:

  void                      noSuchTypeError_

    ( const String&           name )         const;

  void                      noSuchDofError_

    ( idx_t                   iitem,
      idx_t                   itype )        const;

  void                      missingDofError_

    ( const IdxVector&        iitems,
      const IdxVector&        itypes )       const;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


Reordering                  makeDofOrdering

  ( const DofSpace&           dofs,
    const IdxVector&          itemPerm );





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   isDense
//-----------------------------------------------------------------------


inline bool DofSpace::isDense () const
{
  const double  fillFrac =

    (double) dofCount() / ((double) itemCount() *
                           (double) typeCount() + 0.1);

  return (fillFrac > 0.5);
}


//-----------------------------------------------------------------------
//   getTypeIndex
//-----------------------------------------------------------------------


inline idx_t DofSpace::getTypeIndex ( const String& name ) const
{
  idx_t  itype = findType ( name );

  if ( itype < 0 )
  {
    noSuchTypeError_ ( name );
  }

  return itype;
}


//-----------------------------------------------------------------------
//   getDofIndex
//-----------------------------------------------------------------------


inline idx_t DofSpace::getDofIndex

  ( idx_t  iitem,
    idx_t  itype ) const

{
  idx_t  idof = findDofIndex ( iitem, itype );

  if ( idof < 0 )
  {
    noSuchDofError_ ( iitem, itype );
  }

  return idof;
}


//-----------------------------------------------------------------------
//   getDofIndices
//-----------------------------------------------------------------------


inline void DofSpace::getDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  const idx_t  k = findDofIndices ( idofs, iitems, itypes );

  if ( k != iitems.size() * itypes.size() )
  {
    missingDofError_ ( iitems, itypes );
  }
}


JIVE_END_PACKAGE( util )

#endif
