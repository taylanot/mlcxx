
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

#ifndef JIVE_UTIL_XITEMMAP_H
#define JIVE_UTIL_XITEMMAP_H

#include <jive/util/ItemMap.h>
#include <jive/util/StorageMode.h>


JIVE_BEGIN_PACKAGE( util )


class Reordering;


//-----------------------------------------------------------------------
//   class XItemMap
//-----------------------------------------------------------------------


class XItemMap : public ItemMap
{
 public:

  JEM_DECLARE_CLASS   ( XItemMap, ItemMap );


  virtual void          clear       ()       = 0;

  virtual void          reserve

    ( idx_t               count );

  virtual void          trimToSize  ();

  virtual idx_t         addItem     ()       = 0;

  virtual idx_t         addItem

    ( idx_t               itemID )           = 0;

  virtual idx_t         addItems

    ( idx_t               count );

  virtual void          reorderItems

    ( const Reordering&   reord )            = 0;


 protected:

  virtual              ~XItemMap    ();

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


Ref<XItemMap>           newXItemMap

  ( StorageMode           mode = DEFAULT_STORAGE );


JIVE_END_PACKAGE( util )

#endif
