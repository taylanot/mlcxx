
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

#ifndef JIVE_GEOM_SHAPETABLE_H
#define JIVE_GEOM_SHAPETABLE_H

#include <jem/base/NamedObject.h>
#include <jive/Array.h>
#include <jive/util/Topology.h>
#include <jive/util/EventSource.h>
#include <jive/geom/import.h>
#include <jive/geom/ShapeArray.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::util::Event;


//-----------------------------------------------------------------------
//   class ShapeTable
//-----------------------------------------------------------------------


class ShapeTable : public NamedObject,
                   public EventSource
{
 public:

  JEM_DECLARE_CLASS       ( ShapeTable, NamedObject );

  typedef util::TopoVector  TopoVector;


  Event < Self& >           changeEvent;

  static const int          ANY_TYPE;
  static const int          INTERNAL_TYPE;
  static const int          BOUNDARY_TYPE;
  static const int          INTERFACE_TYPE;
  static const int          REGULAR_TYPE;
  static const int          GRAD2_TYPE;
  static const int          BEZIER_TYPE;


  idx_t                     itemCount       () const;
  virtual ItemSet*          getItems        () const = 0;
  virtual idx_t             maxPrecision    () const = 0;
  virtual String            getContext      () const override;
  virtual void              resetEvents     ()       override;

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat );

  virtual void              getConfig

    ( const Properties&       conf )           const;

  virtual ShapeVector       getShapes

    ( const IdxVector&        map,
      idx_t                   prec = 0 )       const;

  virtual ShapeVector       getShapes

    ( const IdxVector&        map,
      const IdxVector&        precs    )       const;

  virtual ShapeMatrix       getShapes

    ( const IdxVector&        map,
      idx_t                   minPrec,
      idx_t                   maxPrec  )       const;

  virtual Shape*            getShapeOf

    ( idx_t                   iitem,
      idx_t                   prec = 0 )       const = 0;

  virtual ShapeVector       getShapesOf

    ( const IdxVector&        map,
      const IdxVector&        iitems,
      idx_t                   prec = 0 )       const;

  virtual ShapeVector       getShapesOf

    ( const IdxVector&        map,
      const IdxVector&        iitems,
      const IdxVector&        precs    )       const;

  virtual ShapeMatrix       getShapesOf

    ( const IdxVector&        map,
      const IdxVector&        iitems,
      idx_t                   minPrec,
      idx_t                   maxPrec )        const = 0;

  TopoVector                getShapeTopos

    ( const IdxVector&        map )            const;

  TopoVector                getShapeToposOf

    ( const IdxVector&        map,
      const IdxVector&        iitems )         const;

  void                      checkShapes

    ( const String&           context,
      int                     flags,
      idx_t                   prec = 0 )       const;

  void                      checkShapes

    ( const String&           context,
      int                     flags,
      const IdxVector&        precs )          const;

  void                      checkShapes

    ( const String&           context,
      int                     flags,
      idx_t                   minPrec,
      idx_t                   maxPrec  )       const;

  void                      checkShapesOf

    ( const String&           context,
      const IdxVector&        iitems,
      int                     flags,
      idx_t                   prec = 0 )       const;

  void                      checkShapesOf

    ( const String&           context,
      const IdxVector&        iitems,
      int                     flags,
      const IdxVector&        precs    )       const;

  void                      checkShapesOf

    ( const String&           context,
      const IdxVector&        iitems,
      int                     flags,
      idx_t                   minPrec,
      idx_t                   maxPrec  )       const;

  static int                getTypeFlags

    ( const Shape&            shape  );

  void                      store

    ( const Properties&       globdat )        const;

  static Ref<Self>          find

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Ref<Self>          get

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat,
      const String&           context );


 protected:

  explicit                  ShapeTable

    ( const String&           name = "" );

  virtual                  ~ShapeTable      ();

  virtual void              emitEvents_

    ( int                     events )               override;


 private:

  void                      noShapeError_

    ( const String&           context,
      idx_t                   iitem    )       const;

  void                      noShapeError_

    ( const String&           context,
      idx_t                   iitem,
      idx_t                   prec     )       const;

  void                      shapeTypeError_

    ( const String&           context,
      const Shape&            shape,
      idx_t                   iitem,
      int                     err      )       const;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


util::TopoVector            getTopologies

  ( const IShapeVector&       shapes );


JIVE_END_PACKAGE( geom )

#endif
