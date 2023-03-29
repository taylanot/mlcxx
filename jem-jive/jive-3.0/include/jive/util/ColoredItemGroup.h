
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

#ifndef JIVE_UTIL_COLOREDITEMGROUP_H
#define JIVE_UTIL_COLOREDITEMGROUP_H

#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jive/util/ItemGroup.h>


namespace jem
{
  namespace util
  {
    class BitSet;
  }
}


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class ColoredItemGroup
//-----------------------------------------------------------------------


class ColoredItemGroup : public ItemGroup,
                         public Clonable,
                         public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( ColoredItemGroup, ItemGroup );


                          ColoredItemGroup  ();

  explicit                ColoredItemGroup

    ( const IdxVector&      colors,
      const Ref<ItemSet>&   items );

                          ColoredItemGroup

    ( const IdxVector&      colors,
      const IdxVector&      iitems,
      const Ref<ItemSet>&   items );

  virtual void            readFrom

    ( ObjectInput&          in  )                    override;

  virtual void            writeTo

    ( ObjectOutput&         out )              const override;

  virtual Ref<Object>     clone             () const override;
  virtual idx_t           size              () const override;
  virtual IdxVector       getIndices        () const override;

  virtual idx_t           getIndex

    ( idx_t                 i )                const override;

  virtual ItemSet*        getItems          () const override;

  virtual bool            contains

    ( idx_t                 iitem )            const override;

  virtual void            filter

    ( const IdxVector&      iitems )           const override;

  idx_t                   listCount         () const;

  IdxVector               getList

    ( idx_t                 ilist )            const;

  idx_t                   getListOffset

    ( idx_t                 ilist )            const;

  idx_t                   getListColor

    ( idx_t                 ilist )            const;



 protected:

  virtual                ~ColoredItemGroup  ();


 private:

  void                    initMask_         () const;
  void                    freeMask_         ();
  void                    connect_          ();

  void                    init_

    ( const IdxVector&      colors,
      const IdxVector&      iitems );

  void                    itemsResized_

    ( idx_t                 size );

  void                    itemsReordered_

    ( const Reordering&     reord );

  static inline void      checkListIndex_

    ( idx_t                 ilist,
      idx_t                 ubound );


 private:

  Ref<ItemSet>            items_;
  IdxVector               iitems_;
  IdxVector               listOffsets_;
  IdxVector               listColors_;

  IdxVector*              imask_;
  jem::util::BitSet*      bmask_;

};


JIVE_END_PACKAGE( util )

#endif
