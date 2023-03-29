
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

#ifndef JEM_GL_LIGHTSNODE_H
#define JEM_GL_LIGHTSNODE_H

#include <jem/gl/Lights.h>
#include <jem/gl/GfxNode.h>


JEM_BEGIN_PACKAGE( gl )


class GLContext;
class GLFunctions1;
class RenderContext;


//-----------------------------------------------------------------------
//   class LightsNode
//-----------------------------------------------------------------------


class LightsNode : public GfxNode
{
 public:

  JEM_DECLARE_CLASS       ( LightsNode, GfxNode );

  static const int          LIGHT0      = Lights::LIGHT0;
  static const int          LIGHT1      = Lights::LIGHT1;
  static const int          LIGHT2      = Lights::LIGHT2;
  static const int          ALL_LIGHTS  = Lights::ALL_LIGHTS;

  static const int          LIGHT_COUNT = Lights::LIGHT_COUNT;


  explicit                  LightsNode

    ( const String&           name = "lights" );

  virtual bool              takeGLAction

    ( RenderContext&          rtx,
      int                     action,
      const Properties&       params )             override;

  virtual void              getVisibility

    ( bool&                   viz,
      Box&                    vbox,
      const Transform&        proj )         const override;

  virtual void              configure

    ( const Properties&       props )              override;

  virtual void              getConfig

    ( const Properties&       props )        const override;

  virtual void              listProps

    ( const Properties&       props )        const override;

  void                      enable

    ( int                     mask,
      bool                    choice = true );

  bool                      isEnabled

    ( int                     idx )          const;

  void                      setColor

    ( int                     mask,
      const Color&            color );

  Color                     getColor

    ( int                     idx )          const;

  void                      setDirection

    ( int                     mask,
      const Vertex3f&         dir );

  Vertex3f                  getDirection

    ( int                     idx )          const;

  void                      setAmbient

    ( const Color&            color );

  Color                     getAmbient    () const noexcept;


 protected:

  virtual                  ~LightsNode    ();


 private:

  struct                    Light_
  {
    Vertex4f                  pos;
    Color                     color;
    int                       glidx;
    bool                      enabled;
  };

  void                      initLights_

    ( GLContext&              gtx );

  void                      initLight_

    ( GLContext&              gtx,
      Light_&                 light,
      GLFunctions1&           funcs );

  static GLFunctions1*      getFunctions_

    ( const GLContext&        gtx );


 private:

  Light_                    lights_ [LIGHT_COUNT];
  Color                     ambient_;
  int                       dirtyBits_;

};


JEM_END_PACKAGE( gl )

#endif
