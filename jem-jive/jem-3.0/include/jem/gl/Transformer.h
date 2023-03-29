
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

#ifndef JEM_GL_TRANSFORMER_H
#define JEM_GL_TRANSFORMER_H

#include <jem/base/Object.h>
#include <jem/gl/Transform.h>


JEM_BEGIN_PACKAGE( gl )


class Orb;
class Rect;
class Camera;
class GfxNode;
class MouseDrag;
class RenderContext;


//-----------------------------------------------------------------------
//   class Transformer
//-----------------------------------------------------------------------


class Transformer : public Object
{
 public:

  static const int        ROT_Z;
  static const int        ROT_XY;
  static const int        TRANS_Z;
  static const int        TRANS_XY;
  static const int        SNAPPY;

  static const int        ROT_MODE;
  static const int        TRANS_MODE;


                          Transformer   ();

  inline int              mode          () const noexcept;
  inline bool             isActive      () const noexcept;
  void                    reset         ();

  void                    cancel

    ( GfxNode&              obj,
      const Camera&         cam );

  void                    init

    ( int                   mode,
      const GfxNode&        obj,
      const Camera&         cam,
      const MouseDrag&      mouse );

  void                    transform

    ( GfxNode&              obj,
      Camera&               cam,
      const MouseDrag&      mouse );

  void                    redraw

    ( const Camera&         cam,
      RenderContext&        rtx )          const;

  static void             align

    ( GfxNode&              obj,
      Camera&               cam );

  static void             alignAxis

    ( int                   axis,
      GfxNode&              obj,
      Camera&               cam );


 protected:

  virtual                ~Transformer   ();


 private:

  void                    initRot_

    ( int                   mode,
      const GfxNode&        obj,
      const Camera&         cam,
      const MouseDrag&      mouse );

  void                    rotate_

    ( GfxNode&              obj,
      Camera&               cam,
      const Vertex3d&       mpos );

  void                    zRotate_

    ( Transform&            t,
      const Vertex3d&       mpos );

  void                    xyRotate_

    ( Transform&            t,
      const Vertex3d&       mpos );

  void                    initTrans_

    ( int                   mode,
      const GfxNode&        obj,
      const Camera&         cam,
      const MouseDrag&      mouse );

  void                    translate_

    ( GfxNode&              obj,
      Camera&               cam,
      const Vertex3d&       mpos );

  void                    xyTranslate_

    ( GLdouble&             dx,
      GLdouble&             dy,
      const Vertex3d&       mpos );


 private:

  Ref<Orb>                orb_;

  int                     mode_;
  int                     zeta_;
  int                     theta_;

  Vertex2d                origin_;
  Vertex2d                objPos_;
  Vertex3d                objCenter_;

  Transform               objTransf_;
  Transform               worldTransf_;

};


//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   mode
//-----------------------------------------------------------------------


inline int Transformer::mode () const noexcept
{
  return mode_;
}


//-----------------------------------------------------------------------
//   isActive
//-----------------------------------------------------------------------


inline bool Transformer::isActive () const noexcept
{
  return ((mode_ & (ROT_MODE | TRANS_MODE)) != 0);
}


JEM_END_PACKAGE( gl )

#endif
