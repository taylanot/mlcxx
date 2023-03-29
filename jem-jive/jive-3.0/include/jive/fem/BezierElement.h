
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

#ifndef JIVE_FEM_BEZIERELEMENT_H
#define JIVE_FEM_BEZIERELEMENT_H

#include <jive/geom/BezierExtractor.h>
#include <jive/fem/InternalElement.h>


namespace jive
{
  namespace geom
  {
    class BezierExtension;
  }
}


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class BezierElement
//-----------------------------------------------------------------------


class BezierElement : public InternalElement
{
 public:

  JEM_DECLARE_CLASS       ( BezierElement, InternalElement );

  typedef
    util::IntDBColumn       IntColumn;
  typedef
    util::FloatDBColumn     FloatColumn;
  typedef
    geom::BezierExtractor   BezierExtractor;
  typedef
    geom::BezierExtension   BezierExtension;


  explicit                  BezierElement

    ( const ElementSet&       elems,
      const Ref<Database>&    dbase,
      const Ref<IShape>&      shape = nullptr );

  virtual void              loadData

    ( idx_t                   ielem )         override;


 protected:

  virtual                  ~BezierElement  ();

  virtual void              setupChanged_

    ( int                     flags )         override;


 private:

  void                      shapeChanged_  ();


 private:

  class                     DBCols_
  {
   public:

    IntColumn*                irows;
    IntColumn*                jcols;
    FloatColumn*              values;
    FloatColumn*              weights;
  };

  Ref<Database>             dbase_;
  DBCols_                   dbCols_;
  BezierExtension*          bezier_;
  BezierExtractor           extractor_;
  Vector                    weights_;

};


JIVE_END_PACKAGE( fem )

#endif
