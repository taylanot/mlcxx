
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

#ifndef JIVE_GL_VECTORFIELD_H
#define JIVE_GL_VECTORFIELD_H

#include <jive/gl/AppRef.h>
#include <jive/gl/DataSource.h>


JIVE_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class VectorField
//-----------------------------------------------------------------------


class VectorField : public Object
{
 public:

  typedef VectorField     Self;
  typedef Object          Super;


                          VectorField

    ( idx_t                 rank,
      const Ref<ItemSet>&   items );

                          VectorField

    ( idx_t                 rank,
      const Ref<ItemSet>&   items,
      const String          compNames[3] );

  void                    configure

    ( const Properties&     props,
      const Properties&     dataSets );

  void                    configure

    ( const String&         name,
      const Properties&     props,
      const Properties&     dataSets );

  void                    getConfig

    ( const Properties&     props )          const;

  void                    getData

    ( double&               scale,
      const Matrix&         vfield,
      const Properties&     globdat,
      const String&         context,
      const AppContext&     apx )            const;

  inline double           getScaleStep    () const;

  void                    setScaleStep

    ( double                step );

  inline double           getPostScale    () const;

  void                    setPostScale

    ( double                scale );

 public:

  bool                    autoScale;


 protected:

  virtual                ~VectorField     ();


 private:

  inline void             initScaling_    ();

  void                    configure_

    ( const String&         str,
      const Properties&     dataSets );

  void                    dataError_

    ( const String&         context,
      const DataSet&        dataSet,
      const AppContext&     apx )            const;


 private:

  idx_t                   rank_;
  AppRef<ItemSet>         items_;

  DataSource              data_[3];
  String                  compNames_[3];

  double                  preScale_;
  double                  postScale_;
  double                  scaleStep_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getScaleStep
//-----------------------------------------------------------------------


inline double VectorField::getScaleStep () const
{
  return scaleStep_;
}


//-----------------------------------------------------------------------
//   getPostScale
//-----------------------------------------------------------------------


inline double VectorField::getPostScale () const
{
  return postScale_;
}


JIVE_END_PACKAGE( gl )

#endif
