
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

#ifndef JEM_GL_POINTSAMPLER_H
#define JEM_GL_POINTSAMPLER_H

#include <jem/base/Time.h>
#include <jem/util/event/Event.h>
#include <jem/gl/input.h>
#include <jem/gl/Color.h>
#include <jem/gl/Vertex.h>
#include <jem/gl/GfxNode.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class PointSampler
//-----------------------------------------------------------------------


class PointSampler : public GfxNode
{
 public:

  JEM_DECLARE_CLASS       ( PointSampler, GfxNode );

  static const double       MIN_INTERVAL;

  struct                    Sample
  {
    MouseClick                mouse;
    Color                     color;
    Vertex3d                  point;
    Vertex3d                  error;
  };

  util::Event
    < RenderContext&,
      const Sample& >       sampleEvent;


                            PointSampler

    ( const String&           name,
      const Ref<GfxNode>&     obj );

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params )             override;

  virtual bool              takeGLAction

    ( RenderContext&          rtx,
      int                     action,
      const Properties&       params )             override;

  virtual void              getVisibility

    ( bool&                   viz,
      Box&                    vbox,
      const Transform&        proj )         const override;

  void                      takeSample

    ( RenderContext&          rtx,
      const MouseClick&       click );

  void                      setTarget

    ( const Ref<GfxNode>&     obj );

  inline GfxNode*           getTarget     () const noexcept;

  void                      setMouseMask

    ( int                     btnMask );

  static Ref<Self>          find

    ( const Ref<GfxNode>&     root,
      const Ref<Object>&      target );


 protected:

  virtual                  ~PointSampler  ();


 private:

  void                      reset_        ();

  static bool               takeSample_

    ( RenderContext&          rtx,
      GLfloat&                depth,
      Sample&                 sample,
      GfxNode&                target );

  static Ref<Self>          getSampler_

    ( const Ref<Object>&      obj );

  static void               sampleCallback_

    ( RenderContext&          rtx,
      const Ref<Object>&      obj );

  static void               click2Callback_

    ( RenderContext&          rtx,
      const Ref<Object>&      obj );


 private:

  static const char*        TARGET_;
  static const char*        SAMPLER_;
  static const char*        GET_SAMPLE_;
  static const char*        FIND_SAMPLER_;


  Ref<GfxNode>              target_;
  int                       btnMask_;
  Sample                    sample_;
  int                       button_;
  GLfloat                   depth_;
  bool                      focus_;
  Time                      lastTime_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getTarget
//-----------------------------------------------------------------------


inline GfxNode* PointSampler::getTarget () const noexcept
{
  return target_.get ();
}


JEM_END_PACKAGE( gl )

#endif
