
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

#ifndef JIVE_UTIL_ITEMSET_H
#define JIVE_UTIL_ITEMSET_H

#include <jem/base/NamedObject.h>
#include <jive/Array.h>
#include <jive/util/import.h>
#include <jive/util/EventSource.h>


JIVE_BEGIN_PACKAGE( util )


class ItemMap;
class Reordering;


//-----------------------------------------------------------------------
//   class ItemSet
//-----------------------------------------------------------------------


class ItemSet : public NamedObject,
                public EventSource
{
 public:

  JEM_DECLARE_CLASS     ( ItemSet, NamedObject );

  Event < Self& >         changeEvent;

  Event
    < idx_t, Self& >      newSizeEvent;
  Event
    < const Reordering&,
      Self& >             newOrderEvent;


  virtual String          getContext    () const override;
  virtual idx_t           size          () const;
  virtual String          getTagName    () const;
  virtual String          getItemName   () const = 0;
  virtual String          getItemsName  () const;
  virtual ItemMap*        getItemMap    () const = 0;

  String                  getItemPath

    ( idx_t                 itemID )       const;

  idx_t                   findItem

    ( idx_t                 itemID )       const;

  idx_t                   getItemID

    ( idx_t                 iitem  )       const;

  virtual void            printSubset

    ( PrintWriter&          out,
      const IdxVector&      iitems )       const = 0;

  void                    printTo

    ( PrintWriter&          out )          const;

  inline void             store

    ( const Properties&     globdat )      const;

  void                    store

    ( const String&         name,
      const Properties&     globdat )      const;

  static Ref<ItemSet>     find

    ( const String&         name,
      const Properties&     globdat );

  static Ref<ItemSet>     get

    ( const String&         name,
      const Properties&     globdat,
      const String&         context );

  static void             erase

    ( const String&         name,
      const Properties&     globdat );

  static Ref<Dict>        getAll

    ( const Properties&     globdat );

  static StringVector     listAll

    ( const Properties&     globdat );



 protected:

  explicit                ItemSet

    ( const String&         name = "" );

  virtual                ~ItemSet       ();

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


inline void ItemSet::store ( const Properties& globdat ) const
{
  store ( myName_, globdat );
}


JIVE_END_PACKAGE( util )

#endif
