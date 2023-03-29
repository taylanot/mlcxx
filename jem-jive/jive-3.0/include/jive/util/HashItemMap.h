
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

#ifndef JIVE_UTIL_HASHITEMMAP_H
#define JIVE_UTIL_HASHITEMMAP_H

#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jem/util/Flex.h>
#include <jive/util/import.h>
#include <jive/util/XItemMap.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class HashItemMap
//-----------------------------------------------------------------------


class HashItemMap : public XItemMap,
                    public Clonable,
                    public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( HashItemMap, XItemMap );


                          HashItemMap    ();

                          HashItemMap

    ( const Self&           rhs );

  virtual void            readFrom

    ( ObjectInput&          in  )                override;

  virtual void            writeTo

    ( ObjectOutput&         out )          const override;

  virtual Ref<Object>     clone         () const override;
  virtual idx_t           size          () const override;
  virtual void            clear         ()       override;

  virtual void            reserve

    ( idx_t                 count )              override;

  virtual void            trimToSize    ()       override;

  virtual idx_t           addItem       ()       override;

  virtual idx_t           addItem

    ( idx_t                 itemID )             override;

  virtual idx_t           addItems

    ( idx_t                 count )              override;

  virtual void            reorderItems

    ( const Reordering&     reord )              override;

  virtual idx_t           findItem

    ( idx_t                 itemID )       const override;

  virtual idx_t           findItems

    ( const IdxVector&      iitems,
      const IdxVector&      itemIDs )      const override;

  virtual idx_t           getItemID

    ( idx_t                 iitem )        const override;

  virtual void            getItemIDs

    ( const IdxVector&      itemIDs,
      const IdxVector&      iitems )       const override;


 protected:

  virtual                ~HashItemMap    ();


 private:

  class                   IndexMap_;

  IndexMap_*              indexMap_;
  jem::util::Flex<idx_t>  itemIDs_;
  idx_t                   nextID_;

};


JIVE_END_PACKAGE( util )

#endif
