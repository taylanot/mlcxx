
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

#ifndef JIVE_UTIL_XPOINTSET_H
#define JIVE_UTIL_XPOINTSET_H

#include <jive/util/PointSet.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class XPointSet
//-----------------------------------------------------------------------


class XPointSet : public PointSet
{
 public:

  JEM_DECLARE_CLASS     ( XPointSet, PointSet );


  virtual void            clear         () = 0;

  virtual void            reserve

    ( idx_t                 count );

  virtual void            trimToSize    ();

  virtual idx_t           addPoint

    ( const Vector&         coords )       = 0;

  virtual idx_t           addPoint

    ( idx_t                 id,
      const Vector&         coords )       = 0;

  virtual idx_t           addPoints

    ( const Matrix&         coords )       = 0;

  void                    erasePoints

    ( const IdxVector&      ipoints );

  virtual void            reorderPoints

    ( const Reordering&     reord   )      = 0;

  virtual void            setPointCoords

    ( idx_t                 ipoint,
      const Vector&         coords )       = 0;

  virtual void            setPointXCoord

    ( idx_t                 ipoint,
      double                coord,
      idx_t                 idim = 0 )     = 0;

  virtual void            setCoords

    ( const Matrix&         coords );

  virtual void            setXCoords

    ( const Vector&         coords,
      idx_t                 idim = 0 );

  virtual void            setSomeCoords

    ( const IdxVector&      ipoints,
      const Matrix&         coords );

  virtual void            setSomeXCoords

    ( const IdxVector&      ipoints,
      const Vector&         coords,
      idx_t                 idim = 0 );


 protected:

                          XPointSet

    ( const String&         name = "" );

  virtual                ~XPointSet     ();

};


JIVE_END_PACKAGE( util )

#endif
