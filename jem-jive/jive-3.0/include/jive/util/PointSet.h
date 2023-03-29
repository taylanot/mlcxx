
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

#ifndef JIVE_UTIL_POINTSET_H
#define JIVE_UTIL_POINTSET_H

#include <jive/util/ItemSet.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class PointSet
//-----------------------------------------------------------------------


class PointSet : public ItemSet
{
 public:

  JEM_DECLARE_CLASS     ( PointSet, ItemSet );

  Event < Self& >         newGeomEvent;
  Event < idx_t, Self& >  newRankEvent;


  virtual idx_t           rank          () const = 0;
  virtual String          getItemName   () const override;

  virtual void            getPointCoords

    ( const Vector&         coords,
      idx_t                 ipoint  )      const = 0;

  virtual double          getPointXCoord

    ( idx_t                 ipoint,
      idx_t                 idim = 0 )     const = 0;

  virtual void            getCoords

    ( const Matrix&         coords )       const;

  virtual void            getXCoords

    ( const Vector&         coords,
      idx_t                 idim = 0 )     const;

  virtual void            getSomeCoords

    ( const Matrix&         coords,
      const IdxVector&      ipoints )      const;

  virtual void            getSomeXCoords

    ( const Vector&         coords,
      const IdxVector&      ipoints,
      idx_t                 idim = 0 )     const;

  virtual Matrix          toMatrix      () const;

  virtual void            printSubset

    ( PrintWriter&          out,
      const IdxVector&      ipoints )      const override;


 protected:

  explicit                PointSet

    ( const String&         name = "" );

  virtual                ~PointSet      ();

};


JIVE_END_PACKAGE( util )

#endif
