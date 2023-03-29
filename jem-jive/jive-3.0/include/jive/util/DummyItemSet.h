
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

#ifndef JIVE_UTIL_DUMMYITEMSET_H
#define JIVE_UTIL_DUMMYITEMSET_H

#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jive/util/ItemSet.h>
#include <jive/util/EventMask.h>
#include <jive/util/StorageMode.h>


JIVE_BEGIN_PACKAGE( util )


class XItemMap;


//-----------------------------------------------------------------------
//   class DummyItemSet
//-----------------------------------------------------------------------


class DummyItemSet : public ItemSet,
                     public Clonable,
                     public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( DummyItemSet, ItemSet );


                          DummyItemSet ();

  explicit                DummyItemSet

    ( const String&         name,
      const String&         item = "item",
      const Ref<XItemMap>&  map  = nullptr );

                          DummyItemSet

    ( const Self&           rhs );

  virtual void            resetEvents   ()       override;

  virtual void            readFrom

    ( ObjectInput&          in  )                override;

  virtual void            writeTo

    ( ObjectOutput&         out )          const override;

  virtual Ref<Object>     clone         () const override;
  virtual idx_t           size          () const override;
  virtual String          getItemName   () const override;
  virtual ItemMap*        getItemMap    () const override;

  virtual void            clear         ();

  void                    reserve

    ( idx_t                 count );

  void                    trimToSize    ();

  idx_t                   addItem       ();

  idx_t                   addItem

    ( idx_t                 itemID );

  idx_t                   addItems

    ( idx_t                 count );

  idx_t                   addItems

    ( const IdxVector&      itemIDs );

  void                    eraseItems

    ( const IdxVector&      iitems );

  void                    reorderItems

    ( const Reordering&     reord );

  virtual void            printSubset

    ( PrintWriter&          out,
      const IdxVector&      iitems )       const override;


 protected:

  virtual                ~DummyItemSet  ();

  virtual void            emitEvents_

    ( int                   events )             override;


 private:

  static const int        NEW_SIZE_;

  Ref<XItemMap>           map_;
  String                  itemName_;
  EventMask               events_;
  idx_t                   itemCount_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


Ref<DummyItemSet>         newDummyItemSet

  ( StorageMode             mode = MINIMAL_STORAGE );

Ref<DummyItemSet>         newDummyItemSet

  ( const String&           name,
    const String&           item = "item",
    StorageMode             mode = MINIMAL_STORAGE );


JIVE_END_PACKAGE( util )

#endif
