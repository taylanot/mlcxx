
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

#ifndef JIVE_GEOM_STDSHAPETABLE_H
#define JIVE_GEOM_STDSHAPETABLE_H

#include <jive/geom/import.h>
#include <jive/geom/ShapeTable.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class StdShapeTable
//-----------------------------------------------------------------------


class StdShapeTable : public ShapeTable
{
 public:

  JEM_DECLARE_CLASS       ( StdShapeTable, ShapeTable );


  virtual void              resetEvents     ()       override;

  virtual ItemSet*          getItems        () const override;
  virtual idx_t             maxPrecision    () const override;

  virtual ShapeVector       getShapes

    ( const IdxVector&        map,
      idx_t                   prec )           const override;

  virtual ShapeMatrix       getShapes

    ( const IdxVector&        map,
      idx_t                   minPrec,
      idx_t                   maxPrec  )       const override;

  virtual Shape*            getShapeOf

    ( idx_t                   iitem,
      idx_t                   prec )           const override;

  virtual ShapeVector       getShapesOf

    ( const IdxVector&        shapeMap,
      const IdxVector&        iitems,
      idx_t                   prec )           const override;

  virtual ShapeMatrix       getShapesOf

    ( const IdxVector&        shapeMap,
      const IdxVector&        iitems,
      idx_t                   minPrec,
      idx_t                   maxPrec )        const override;


 protected:

                            StdShapeTable   ();

                            StdShapeTable

    ( const String&           name,
      const Ref<ItemSet>&     items,
      idx_t                   maxPrec = 0 );

  virtual                  ~StdShapeTable   ();

  virtual void              update_         ();
  void                      connect_        ();

  void                      itemsResized_

    ( idx_t                   itemCount );

  void                      itemsReordered_

    ( const Reordering&       reord );


 private:

  void                      checkItemIndex_

    ( idx_t                   iitem )          const;


 protected:

  Ref<ItemSet>              items_;
  ShapeMatrix               shapes_;
  IdxVector                 map_;

};


JIVE_END_PACKAGE( geom )

#endif
