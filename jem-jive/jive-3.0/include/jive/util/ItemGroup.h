
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

#ifndef JIVE_UTIL_ITEMGROUP_H
#define JIVE_UTIL_ITEMGROUP_H

#include <jem/base/Object.h>
#include <jive/Array.h>
#include <jive/util/import.h>
#include <jive/util/EventSource.h>


JIVE_BEGIN_PACKAGE( util )


class ItemSet;
class Reordering;


//-----------------------------------------------------------------------
//   class ItemGroup
//-----------------------------------------------------------------------


class ItemGroup : public Object,
                  public EventSource
{
 public:

  JEM_DECLARE_CLASS       ( ItemGroup, Object );

  static const int          GROUP_REORDERED;
  static const int          ITEMS_REORDERED;

  Event
    < idx_t, Self& >        newSizeEvent;
  Event
    < const Reordering&,
      int, Self& >          newOrderEvent;


  virtual void              resetEvents   ()       override;
  virtual idx_t             size          () const = 0;
  virtual IdxVector         getIDs        () const;
  virtual IdxVector         getIndices    () const = 0;

  virtual idx_t             getIndex

    ( idx_t                   i )            const;

  virtual ItemSet*          getItems      () const = 0;

  virtual bool              contains

    ( idx_t                   iitem )        const;

  virtual void              filter

    ( const IdxVector&        iitems )       const;

  idx_t                     findMembers

    ( const IdxVector&        jitems,
      const IdxVector&        iitems )       const;

  idx_t                     findNonMembers

    ( const IdxVector&        jitems,
      const IdxVector&        iitems )       const;

  static IdxVector          getUnion

    ( const ItemGroup&        first,
      const ItemGroup&        second );

  static IdxVector          getIntersection

    ( const ItemGroup&        first,
      const ItemGroup&        second );

  inline void               printTo

    ( PrintWriter&            out )          const;

  static void               printGroup

    ( PrintWriter&            out,
      const ItemSet&          items,
      const IdxVector&        iitems );

  static void               printSome

    ( PrintWriter&            out,
      const Dictionary&       groups,
      const String&           tag = "" );

  static void               printAll

    ( PrintWriter&            out,
      const Properties&       globdat );

  String                    getTagName    () const;

  static String             getTagName

    ( const String&           itemName );

  void                      store

    ( const String&           name,
      const Properties&       globdat )      const;

  static Ref<ItemGroup>     find

    ( const String&           name,
      const String&           items,
      const Properties&       globdat );

  static Ref<ItemGroup>     find

    ( const String&           name,
      const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Ref<ItemGroup>     get

    ( const String&           name,
      const String&           items,
      const Properties&       globdat,
      const String&           context );

  static Ref<ItemGroup>     get

    ( const String&           name,
      const Ref<ItemSet>&     items,
      const Properties&       globdat,
      const String&           context );

  static Ref<Dictionary>    findFor

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Ref<Dictionary>    getFor

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );


 protected:

  virtual                  ~ItemGroup     ();

  virtual void              emitEvents_

    ( int                     events )             override;

};






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


inline void ItemGroup::printTo ( PrintWriter& out ) const
{
  printGroup ( out, * getItems(), getIndices() );
}


JIVE_END_PACKAGE( util )

#endif
