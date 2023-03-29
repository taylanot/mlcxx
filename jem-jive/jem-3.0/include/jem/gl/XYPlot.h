
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

#ifndef JEM_GL_XYPLOT_H
#define JEM_GL_XYPLOT_H

#include <jem/util/ObjFlex.h>
#include <jem/gl/import.h>
#include <jem/gl/Size.h>
#include <jem/gl/Array.h>
#include <jem/gl/Color.h>
#include <jem/gl/Vertex.h>
#include <jem/gl/LeafNode.h>
#include <jem/gl/PlotOptions.h>


JEM_BEGIN_PACKAGE( gl )


class Font;
class GLContext;
class MarkerStyle;
class TextProjector;


//-----------------------------------------------------------------------
//   class XYPlot
//-----------------------------------------------------------------------


class XYPlot : public LeafNode, public PlotOptions
{
 public:

  JEM_DECLARE_CLASS       ( XYPlot, LeafNode );

  typedef Tuple<double,2>   Range;

  class                     DataSet;


  explicit                  XYPlot

    ( const String&           name );

                            XYPlot

    ( const String&           name,
      const Size2f&           size );

  virtual void              getBBox

    ( Box&                    box )            const override;

  virtual void              configure

    ( const Properties&       props )                override;

  virtual void              getConfig

    ( const Properties&       props )          const override;

  virtual void              listProps

    ( const Properties&       props )          const override;

  virtual void              redraw

    ( RenderContext&          rtx )                  override;

  virtual void              redrawFast

    ( RenderContext&          rtx,
      double                  lod )                  override;

  virtual bool              exeCommand

    ( RenderContext&          rtx,
      const String&           cmd,
      const Properties&       params )               override;

  virtual void              listOptions

    ( Menu&                   menu )           const override;

  virtual bool              setFont         ()       override;

  void                      clear           ();

  DataSet*                  newDataSet

    ( const String&           key   = "",
      Style                   style = LINES );

  DataSet*                  newDataSet

    ( const String&           key,
      const Array<double>&    xData,
      const Array<double>&    yData,
      Style                   style = LINES );

  bool                      eraseDataSet

    ( DataSet*                dset );

  DataSet*                  findDataSet

    ( const String&           key )            const noexcept;

  Array<DataSet*>           getDataSets     () const;

  inline Size2f             size            () const noexcept;

  void                      resize

    ( const Size2f&           sz );

  void                      setOption

    ( Option                  option,
      bool                    choice = true );

  void                      setOptions

    ( Options                 options );

  void                      toggleOption

    ( Option                  option );

  inline Options            getOptions      () const noexcept;

  void                      setKeyPos

    ( GLfloat                 xpos,
      GLfloat                 ypos );

  inline GLfloat            getKeyXPos      () const noexcept;
  inline GLfloat            getKeyYPos      () const noexcept;

  void                      setXRange

    ( const Range&            range );

  void                      setYRange

    ( const Range&            range );

  Range                     getXRange       () const;
  Range                     getYRange       () const;
  void                      setAutoScale    ();
  void                      setXAutoScale   ();
  void                      setYAutoScale   ();
  inline bool               getXAutoScale   () const noexcept;
  inline bool               getYAutoScale   () const noexcept;

  void                      setScaleStep

    ( double                  step );

  inline double             getScaleStep    () const noexcept;

  Vertex2d                  mapPoint

    ( const Vertex2d&         p )              const;

  inline void               adjustRange

    ( Range&                  r )              const;


 public:

  String                    title;
  String                    xLabel;
  String                    yLabel;


 protected:

  virtual                  ~XYPlot          ();


 private:

  void                      init_

    ( const Size2f&           sz );

  void                      update_         ();

  void                      drawData_

    ( GLContext&              gtx,
      const Box&              bbox )           const;

  void                      drawFrame_

    ( GLContext&              gtx )            const;

  void                      drawXTicks_

    ( GLContext&              gtx,
      TextProjector&          txp,
      int                     count,
      const GLfloat           psize[2],
      const GLfloat           space[3] );

  void                      drawYTicks_

    ( GLContext&              gtx,
      TextProjector&          txp,
      int                     count,
      const GLfloat           psize[2],
      const GLfloat           space[3] );

  void                      drawLabels_

    ( GLContext&              gtx,
      TextProjector&          txp,
      const GLfloat           psize[2],
      const GLfloat           space[3] );

  void                      drawKey_

    ( GLContext&              gtx,
      TextProjector&          txp,
      const Box&              bbox,
      const GLfloat           psize[2],
      const GLfloat           space[3] );

  void                      getPlotSize_

    ( GLContext&              gtx,
      GLfloat                 psize[2] )       const;

  void                      setFontSize_

    ( Font&                   font,
      const GLfloat           psize[2] )       const;

  void                      setPlotRect_    ();

  void                      setPlotRect_

    ( int                     ticks[2],
      const GLfloat           psize[2],
      const GLfloat           space[3],
      const Font&             font );

  void                      setRanges_

    ( const Properties&       props );

  static void               adjustRange_

    ( Range&                  range,
      double                  step = 0.0 );


 private:

  static const char*        TICK_FORMAT_;

  class                     DataSet_;
  friend class              DataSet_;

  class                     Scratch_
  {
   public:

    inline GLfloat*           alloc

      ( idx_t                   count );

   private:

    GLfloatVector             store_;
  };


 private:

  util::ObjFlex             dataSets_;
  Options                   options_;
  GLfloat                   width_;
  GLfloat                   height_;
  GLfloat                   keyPos_[2];
  GLfloat                   bbox_ [4];
  GLfloat                   prect_[4];
  GLfloat                   depth_;
  Range                     xRange_;
  Range                     yRange_;
  bool                      xAutoScale_;
  bool                      yAutoScale_;
  double                    scaleStep_;
  bool                      updated_;
  int                       nextStyle_;

  Scratch_                  scratch_;

};


//-----------------------------------------------------------------------
//   class XYPlot::DataSet
//-----------------------------------------------------------------------


class XYPlot::DataSet
{
 public:

  typedef MarkerStyle       PntStyle;


  void                      clear           ();

  virtual void              setPoints

    ( const Array<double>&    x,
      const Array<double>&    y )                    = 0;

  virtual void              addPoint

    ( double                  x,
      double                  y )                    = 0;

  virtual void              addPoints

    ( const Array<double>&    x,
      const Array<double>&    y )                    = 0;

  virtual idx_t             pointCount      () const = 0;

  virtual void              setMaxPoints

    ( idx_t                   count )                = 0;

  inline idx_t              getMaxPoints    () const noexcept;

  void                      setStyle

    ( GLfloat                 lwidth,
      const Color&            lcolor,
      const Ref<PntStyle>&    pstyle );

  void                      setLineWidth

    ( GLfloat                 width );

  inline GLfloat            getLineWidth    () const noexcept;

  void                      setLineColor

    ( const Color&            color );

  inline Color              getLineColor    () const noexcept;

  void                      setPointStyle

    ( const Ref<PntStyle>&    style );

  inline PntStyle*          getPointStyle   () const noexcept;

  void                      setBaseNumber

    ( idx_t                   nr0 );

  inline idx_t              getBaseNumber   () const noexcept;

  void                      configure

    ( const Properties&       props );

  void                      getConfig

    ( const Properties&       conf )           const;

  void                      listProps

    ( const Properties&       conf )           const;


 public:

  String                    key;


 protected:

                            DataSet         ();
  virtual                  ~DataSet         ();


 protected:

  GLfloat                   lnWidth_;
  Color                     lnColor_;
  Ref<PntStyle>             ptStyle_;
  idx_t                     maxPoints_;
  idx_t                     baseNumber_;

};


//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class XYPlot
//=======================================================================

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline Size2f XYPlot::size () const noexcept
{
  return Size2f( width_, height_ );
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


inline XYPlot::Options XYPlot::getOptions () const noexcept
{
  return options_;
}


//-----------------------------------------------------------------------
//   getKey[X|Y]Pos
//-----------------------------------------------------------------------


inline GLfloat XYPlot::getKeyXPos () const noexcept
{
  return keyPos_[0];
}


inline GLfloat XYPlot::getKeyYPos () const noexcept
{
  return keyPos_[1];
}


//-----------------------------------------------------------------------
//   get[X|Y]AutoScale
//-----------------------------------------------------------------------


inline bool XYPlot::getXAutoScale () const noexcept
{
  return xAutoScale_;
}


inline bool XYPlot::getYAutoScale () const noexcept
{
  return yAutoScale_;
}


//-----------------------------------------------------------------------
//   getScaleStep
//-----------------------------------------------------------------------


inline double XYPlot::getScaleStep () const noexcept
{
  return scaleStep_;
}


//-----------------------------------------------------------------------
//   adjustRange
//-----------------------------------------------------------------------


inline void XYPlot::adjustRange ( Range& r ) const
{
  adjustRange_ ( r, scaleStep_ );
}


//=======================================================================
//   class XYPlot::DataSet
//=======================================================================

//-----------------------------------------------------------------------
//   getMaxPoints
//-----------------------------------------------------------------------


inline idx_t XYPlot::DataSet::getMaxPoints () const noexcept
{
  return maxPoints_;
}


//-----------------------------------------------------------------------
//   getLineWidth
//-----------------------------------------------------------------------


inline GLfloat XYPlot::DataSet::getLineWidth () const noexcept
{
  return lnWidth_;
}


//-----------------------------------------------------------------------
//   getLineColor
//-----------------------------------------------------------------------


inline Color XYPlot::DataSet::getLineColor () const noexcept
{
  return lnColor_;
}


//-----------------------------------------------------------------------
//   getPointStyle
//-----------------------------------------------------------------------


inline MarkerStyle* XYPlot::DataSet::getPointStyle () const noexcept
{
  return ptStyle_.get ();
}


//-----------------------------------------------------------------------
//   getBaseNumber
//-----------------------------------------------------------------------


inline idx_t XYPlot::DataSet::getBaseNumber () const noexcept
{
  return baseNumber_;
}


JEM_END_PACKAGE( gl )

#endif
