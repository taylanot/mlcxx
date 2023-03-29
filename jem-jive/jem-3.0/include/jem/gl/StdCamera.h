
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

#ifndef JEM_GL_STDCAMERA_H
#define JEM_GL_STDCAMERA_H

#include <jem/base/Time.h>
#include <jem/base/Flags.h>
#include <jem/gl/Rect.h>
#include <jem/gl/Camera.h>
#include <jem/gl/Pivotable.h>
#include <jem/gl/Transformable.h>


JEM_BEGIN_PACKAGE( gl )


class GLContext;


//-----------------------------------------------------------------------
//   class StdCamera
//-----------------------------------------------------------------------


class StdCamera : public Camera,
                  public Pivotable,
                  public Transformable
{
 public:

  JEM_DECLARE_CLASS       ( StdCamera, Camera );

  static const int          MAX_QUALITY;

  enum                      Option
  {
                              AUTO_QUALITY = 1 << 0
  };

  typedef Flags<Option>     Options;


  explicit                  StdCamera

    ( const String&           name  = "camera",
      const Ref<GfxNode>&     child = nullptr );

  virtual void              setChild

    ( const Ref<GfxNode>&     child )                override;

  virtual Ref<GfxNode>      getChild        () const override;

  virtual GfxNode*          findNode

    ( const String&           name )           const override;

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params )               override;

  virtual bool              takeGLAction

    ( RenderContext&          rtx,
      int                     action,
      const Properties&       params )               override;

  virtual void              getBBox

    ( Box&                    box )            const override;

  virtual void              getBareBBox

    ( Box&                    box )            const override;

  virtual void              getVisibility

    ( bool&                   viz,
      Box&                    vbox,
      const Transform&        proj )           const override;

  virtual void              configure

    ( const Properties&       props )                override;

  virtual void              getConfig

    ( const Properties&       props )          const override;

  virtual void              listProps

    ( const Properties&       props )          const override;

  virtual Transform         getTransform    () const override;

  virtual void              setTransform

    ( const Transform&        tr )                   override;

  virtual Rect              getViewport     () const override;

  virtual Transform         getProjection   () const override;

  virtual void              reset           ()       override;

  virtual void              translateObj

    ( GfxNode&                node,
      GLdouble                dx,
      GLdouble                dy,
      GLdouble                dz )             const override;

  virtual GLdouble          setZoom

    ( GLdouble                factor )               override;

  virtual GLdouble          getZoom         () const override;

  int                       setQuality

    ( int                     level );

  int                       getQuality      () const noexcept;

  void                      setOption

    ( Option                  option,
      bool                    yesno = true );

  void                      setOptions

    ( Options                 options );

  Options                   getOptions      () const noexcept;


 protected:

  virtual                  ~StdCamera       ();

  virtual Transform         newProjection_  () const = 0;


 protected:

  Ref<GfxNode>              child_;

  Transform                 proj_;
  Transform                 transfm_;

  GLdouble                  zoom_;
  Rect                      vport_;
  Options                   options_;
  bool                      newProj_;


 private:

  void                      redrawFSAA_

    ( RenderContext&          rtx,
      const Properties&       params );

  void                      adjustQuality_

    ( double                  dt );

  void                      drawScreenQuad_

    ( GLContext&              gtx,
      GLuint                  texID,
      GLfloat                 accf,
      int                     level );

 private:

  static const GLfloat      JITTER_[];
  static const int          JINDEX_[];

  class                     Data_;
  class                     FSAAScope_;

  Ref<Data_>                data_;

  int                       quality_;
  int                       maxQuality_;
  double                    nextQuality_;
  Time                      lastLowered_;

};


JEM_DEFINE_FLAG_OPS( StdCamera::Options )


JEM_END_PACKAGE( gl )

#endif
