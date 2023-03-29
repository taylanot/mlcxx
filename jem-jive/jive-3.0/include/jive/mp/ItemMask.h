
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

#ifndef JIVE_MP_ITEMMASK_H
#define JIVE_MP_ITEMMASK_H

#include <jem/base/Object.h>
#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jive/Array.h>
#include <jive/util/EventMask.h>
#include <jive/mp/import.h>


JIVE_BEGIN_PACKAGE( mp )


using jem::util::Event;


//-----------------------------------------------------------------------
//   class ItemMask
//-----------------------------------------------------------------------


class ItemMask : public Object,
                 public EventSource,
                 public Clonable,
                 public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( ItemMask, Object );

  Event < Self& >           newValuesEvent;


                            ItemMask        ();

  explicit                  ItemMask

    ( const Ref<ItemSet>&     items );

                            ItemMask

    ( const Self&             rhs );

  virtual void              resetEvents     ()       override;

  virtual void              readFrom

    ( ObjectInput&            in  )                  override;

  virtual void              writeTo

    ( ObjectOutput&           out )            const override;

  virtual Ref<Object>       clone           () const override;

  idx_t                     size            () const;
  ItemSet*                  getItems        () const;
  BoolVector                toVector        () const;

  void                      clear           ();

  bool                      getValue

    ( idx_t                   iitem )          const;

  void                      getSomeValues

    ( const BoolVector&       values,
      const IdxVector&        iitems )         const;

  idx_t                     getAllValues

    ( const BoolVector&       values )         const;

  void                      setValue

    ( idx_t                   iitem,
      bool                    value );

  void                      setSomeValues

    ( const IdxVector&        iitems,
      const BoolVector&       values );

  void                      setAllValues

    ( const BoolVector&       values );

  IdxVector                 getItemsFor

    ( bool                    value )          const;

  void                      store

    ( const Properties&       globdat )        const;

  static Ref<ItemMask>      find

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Ref<ItemMask>      get

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );


 protected:

  virtual                  ~ItemMask        ();

  virtual void              emitEvents_

    ( int                     events )               override;


 private:

  idx_t                     resizeMask_     ();
  void                      connect_        ();

  void                      itemsResized_

    ( idx_t                   itemCount );

  void                      itemsReordered_

    ( const Reordering&       reord );

  inline void               checkItemIndex_

    ( idx_t                   iitem );

  inline bool               checkItemIndex_

    ( idx_t                   iitem )          const;

  void                      checkIndexAgain_

    ( idx_t                   iitem );


 private:

  static const int          NEW_VALUES_;

  Ref<ItemSet>              items_;
  BoolVector                mask_;
  EventMask                 events_;

};


JIVE_END_PACKAGE( mp )

#endif
