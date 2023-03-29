
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

#ifndef JIVE_UTIL_SPARSEDOFSPACE_H
#define JIVE_UTIL_SPARSEDOFSPACE_H

#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jive/util/XDofSpace.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class SparseDofSpace
//-----------------------------------------------------------------------


class SparseDofSpace : public XDofSpace,
                       public Clonable,
                       public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( SparseDofSpace, XDofSpace );


                            SparseDofSpace ();

  explicit                  SparseDofSpace

    ( const Ref<ItemSet>&     items );

                            SparseDofSpace

    ( const Self&             rhs );

  virtual void              resetEvents     ()       override;

  virtual void              readFrom

    ( ObjectInput&            in  )                  override;

  virtual void              writeTo

    ( ObjectOutput&           out )            const override;

  virtual Ref<Object>       clone           () const override;
  virtual idx_t             typeCount       () const override;
  virtual idx_t             dofCount        () const override;
  virtual void              clearDofs       ()       override;
  virtual void              clearAll        ()       override;

  virtual void              reserve

    ( idx_t                   dofCount )             override;

  virtual void              trimToSize      ()       override;

  virtual idx_t             addType

    ( const String&           name )                 override;

  virtual idx_t             findType

    ( const String&           name )           const override;

  virtual String            getTypeName

    ( idx_t                   itype )          const override;

  virtual StringVector      getTypeNames    () const override;

  virtual idx_t             addDof

    ( idx_t                   iitem,
      idx_t                   itype )                override;

  virtual void              addDofs

    ( const IdxVector&        iitems,
      const IdxVector&        itypes )               override;

  virtual void              reorderDofs

    ( const Reordering&       reord )                override;

  virtual idx_t             findDofIndex

    ( idx_t                   iitem,
      idx_t                   itype  )         const override;

  virtual idx_t             findDofIndices

    ( const IdxVector&        idofs,
      const IdxVector&        iitems,
      const IdxVector&        itypes )         const override;

  virtual idx_t             collectDofIndices

    ( const IdxVector&        idofs,
      const IdxVector&        iitems,
      const IdxVector&        itypes )         const override;

  virtual idx_t             getDofsForItem

    ( const IdxVector&        idofs,
      const IdxVector&        itypes,
      idx_t                   iitem )          const override;

  virtual idx_t             getDofsForType

    ( const IdxVector&        idofs,
      const IdxVector&        iitems,
      idx_t                   itype )          const override;

  virtual void              decodeDofIndex

    ( idx_t&                  iitem,
      idx_t&                  itype,
      idx_t                   idof   )         const override;

  virtual void              decodeDofIndices

    ( const IdxVector&        iitems,
      const IdxVector&        itypes,
      const IdxVector&        idofs  )         const override;

  virtual SparseIdxMatrix   toMatrix        () const override;
  virtual ItemSet*          getItems        () const override;

  void                      setAllDofs

    ( const IdxVector&        iitems,
      const IdxVector&        itypes );

  void                      getAllDofs

    ( const IdxVector&        iitems,
      const IdxVector&        itypes )         const;


 protected:

  virtual                  ~SparseDofSpace  ();

  virtual void              emitEvents_

    ( int                     events )               override;


 private:

  void                      connect_        ();

  void                      itemsResized_

    ( idx_t                   itemCount );

  void                      itemsReordered_

    ( const Reordering&       reord );


 private:

  class                     Data_;

  Ref<Data_>                data_;

};


JIVE_END_PACKAGE( util )

#endif
