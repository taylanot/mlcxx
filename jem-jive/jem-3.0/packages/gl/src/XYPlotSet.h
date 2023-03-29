
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */

#ifndef JEM_GL_XYPLOTSET_H
#define JEM_GL_XYPLOTSET_H

#include <jem/util/Flex.h>
#include <jem/gl/XYPlot.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class XYPlot::DataSet_
//-----------------------------------------------------------------------


class XYPlot::DataSet_ : public Object, public DataSet
{
 public:

  explicit inline           DataSet_

    ( XYPlot*                 plot );

  virtual void              setPoints

    ( const Array<double>&    x,
      const Array<double>&    y )                     override;

  virtual void              addPoint

    ( double                  x,
      double                  y )                     override;

  virtual void              addPoints

    ( const Array<double>&    x,
      const Array<double>&    y )                     override;

  virtual idx_t              pointCount      () const override;

  virtual void              setMaxPoints

    ( idx_t                    count )                override;

  inline Range              getXRange       () const;

  Range                     getXRange

    ( const Range&            yRange )         const;

  inline Range              getYRange       () const;

  Range                     getYRange

    ( const Range&            xRange )         const;

  void                      sieve

    ( const Range&            xRange,
      const Range&            yRange );

  void                      drawFast

    ( GLContext&              gtx,
      const double            scale[2],
      const double            bias [2],
      double                  lod )            const;

  void                      drawLines

    ( GLContext&              gtx,
      const double            scale[2],
      const double            bias [2] )       const;

  void                      drawMarkers

    ( GLContext&              gtx,
      const double            scale[2],
      const double            bias [2],
      const Box&              bbox )           const;


 private:

  void                      update_         ();

  Range                     getRange_

    ( const Range&            vRange,
      const Array<double>&    uData,
      const Array<double>&    vData )          const;


 private:

  XYPlot*                   plot_;
  idx_t                     size_;
  idx_t                     first_;
  idx_t                     last_;
  Array<double>             xData_;
  Array<double>             yData_;
  Range                     xRange_;
  Range                     yRange_;
  util::Flex<idx_t>         pointList_;

};




//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class XYPlot::DataSet_
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline XYPlot::DataSet_::DataSet_ ( XYPlot* p ) :

  plot_ ( p )

{
  size_   = 0_idx;
  first_  = 0_idx;
  last_   = 0_idx;
  xRange_ = 0.0;
  yRange_ = 0.0;
}


//-----------------------------------------------------------------------
//   getXRange
//-----------------------------------------------------------------------


inline XYPlot::Range XYPlot::DataSet_::getXRange () const
{
  return xRange_;
}


//-----------------------------------------------------------------------
//   getYRange
//-----------------------------------------------------------------------


inline XYPlot::Range XYPlot::DataSet_::getYRange () const
{
  return yRange_;
}


//=======================================================================
//   class XYPlot::Scratch_
//=======================================================================


inline GLfloat* XYPlot::Scratch_::alloc ( idx_t count )
{
  if ( store_.size() < count )
  {
    store_.resize ( count );
  }

  return store_.addr ();
}


JEM_END_PACKAGE( gl )

#endif
