
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

#ifndef JEM_GL_GRAPHVIEWER_H
#define JEM_GL_GRAPHVIEWER_H

#include <jem/base/array/Array.h>
#include <jem/util/ObjFlex.h>
#include <jem/gl/Color.h>
#include <jem/gl/PlotOptions.h>


JEM_BEGIN_PACKAGE( gl )


class Rect;
class Viewer;
class Window;
class XYPlot;
class KeyPress;
class MouseDrag;
class MouseClick;
class StdCamera;
class RubberBand;
class MarkerStyle;
class TextBalloon;
class FrameGrabber;
class RenderContext;


//-----------------------------------------------------------------------
//   class GraphViewer
//-----------------------------------------------------------------------


class GraphViewer : public Object, public PlotOptions
{
 public:

  JEM_DECLARE_CLASS       ( GraphViewer, Object );

  typedef Tuple<double,2>   Range;

  class                     DataSet;


  explicit                  GraphViewer

    ( const Ref<Viewer>&      viewer );

  static Ref<Self>          open

    ( const String&           name );

  static Ref<Self>          open

    ( const String&           name,
      int&                    argc,
      char**                  argv );

  void                      freeze          ();
  void                      resume          ();
  void                      replot          ();
  void                      clear           ();
  bool                      shouldClose     ();

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

  void                      setQuality

    ( int                     level );

  int                       getQuality      () const;

  void                      setOption

    ( Option                  options,
      bool                    choice = true );

  void                      setOptions

    ( Options                 options );

  Options                   getOptions      () const;

  void                      setTitle

    ( const String&           title );

  String                    getTitle        () const;

  void                      setLabels

    ( const String&           xLabel,
      const String&           yLabel );

  String                    getXLabel       () const;
  String                    getYLabel       () const;

  void                      setKeyPos

    ( GLfloat                 xpos,
      GLfloat                 ypos );

  GLfloat                   getKeyXPos      () const;
  GLfloat                   getKeyYPos      () const;

  void                      setBgColor

    ( const Color&            color );

  Color                     getBgColor      () const;

  void                      setXRange

    ( const Range&            range );

  void                      setXRange

    ( double                  lo,
      double                  hi );

  void                      setYRange

    ( const Range&            range );

  void                      setYRange

    ( double                  lo,
      double                  hi );

  Range                     getXRange       () const;
  Range                     getYRange       () const;

  void                      setAutoScale    ();
  void                      setXAutoScale   ();
  void                      setYAutoScale   ();

  bool                      getXAutoScale   () const;
  bool                      getYAutoScale   () const;

  void                      setScaleStep

    ( double                  step );

  double                    getScaleStep    () const;

  String                    saveSnapshot

    ( const String&           fname )          const;

  void                      saveSnapshot

    ( OutputStream&           output,
      FrameGrabber&           grabber )        const;


 protected:

  virtual                  ~GraphViewer     ();


 private:

  void                      keyPressed_

    ( RenderContext&          rtx,
      const KeyPress&         key );

  void                      mouseClicked_

    ( RenderContext&          rtx,
      const MouseClick&       mouse );

  void                      mouseDragged_

    ( RenderContext&          rtx,
      const MouseDrag&        mouse );

  void                      exeCommand_

    ( RenderContext&          rtx,
      const String&           cmd );

  void                      rectSelected_

    ( RenderContext&          rtx,
      const Rect&             rect );

  void                      showHelp_       ();

  void                      showMenu_

    ( RenderContext&          rtx );

  void                      showMousePos_

    ( RenderContext&          rtx,
      int                     xPos,
      int                     yPos );

  void                      resizeHandler_

    ( RenderContext&          rtx,
      const Rect&             rect );


 private:

  class                     ReadLock_;
  class                     WriteLock_;
  class                     DataSet_;

  friend class              ReadLock_;
  friend class              WriteLock_;
  friend class              DataSet_;

  Ref<Viewer>               viewer_;
  Ref<Window>               window_;
  Ref<StdCamera>            camera_;
  Ref<XYPlot>               plot_;
  Ref<TextBalloon>          balloon_;
  Ref<RubberBand>           rubber_;

  util::ObjFlex             dataSets_;

  int                       frozen_;
  bool                      changed_;

};


//-----------------------------------------------------------------------
//   class GraphViewer::DataSet
//-----------------------------------------------------------------------


class GraphViewer::DataSet
{
 public:

  typedef MarkerStyle       PntStyle;


  virtual void              setKey

    ( const String&           key )                  = 0;

  virtual String            getKey          () const = 0;

  virtual void              setPoints

    ( const Array<double>&    x,
      const Array<double>&    y )                    = 0;

  virtual void              addPoint

    ( double                  x,
      double                  y )                    = 0;

  virtual void              addPoints

    ( const Array<double>&    x,
      const Array<double>&    y )                    = 0;

  virtual void              setMaxPoints

    ( idx_t                   n )                    = 0;

  virtual idx_t             getMaxPoints    () const = 0;

  virtual void              setStyle

    ( GLfloat                 lwidth,
      const Color&            lcolor,
      const Ref<PntStyle>&    pstyle )               = 0;

  virtual void              setLineWidth

    ( GLfloat                 width )                = 0;

  virtual GLfloat           getLineWidth    () const = 0;

  virtual void              setLineColor

    ( const Color&            color )                = 0;

  virtual Color             getLineColor    () const = 0;

  virtual void              setPointStyle

    ( const Ref<PntStyle>&    style )                = 0;

  virtual PntStyle*         getPointStyle   () const = 0;


 protected:

                            DataSet         ();
  virtual                  ~DataSet         ();

};


JEM_END_PACKAGE( gl )

#endif
