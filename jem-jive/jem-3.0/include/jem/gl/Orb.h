
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

#ifndef JEM_GL_ORB_H
#define JEM_GL_ORB_H

#include <jem/base/Object.h>
#include <jem/gl/Array.h>


JEM_BEGIN_PACKAGE( gl )


class Color;
class GLContext;


//-----------------------------------------------------------------------
//   class Orb
//-----------------------------------------------------------------------


class Orb : public Object
{
 public:

  JEM_DECLARE_CLASS     ( Orb, Object );

  static const int        PART0;
  static const int        PART1;
  static const int        PART2;
  static const int        PART3;
  static const int        PART4;
  static const int        PART5;
  static const int        PART6;
  static const int        PART7;

  static const int        DOME0;
  static const int        DOME1;
  static const int        DOME2;
  static const int        DOME3;
  static const int        DOME4;
  static const int        DOME5;

  static const int        EVEN_PARTS;
  static const int        ODD_PARTS;

  static const int        ALL_PARTS;
  static const int        MAX_QUALITY;


  explicit                Orb

    ( int                   quality = 2 );

  inline int              getQuality      ()     const noexcept;

  void                    setQuality

    ( int                   quality );

  int                     changeQuality

    ( int                   delta );

  void                    drawOrb

    ( GLContext&            gtx,
      GLfloat               radius,
      int                   parts = ALL_PARTS )  const;

  void                    drawOrb

    ( GLContext&            gtx,
      GLfloat               radius,
      const Color&          color,
      int                   parts = ALL_PARTS )  const;

  void                    drawOrb

    ( GLContext&            gtx,
      GLfloat               radius,
      const Color&          color1,
      const Color&          color2,
      int                   parts = ALL_PARTS )  const;

  void                    drawUnitOrb

    ( GLContext&            gtx,
      int                   parts = ALL_PARTS )  const;


 protected:

  virtual                ~Orb             ();


 private:

  class                   Edge_;
  class                   Patch_;
  friend class            Patch_;

  struct                  Mesh_
  {
    bool                    meshed;
    int                     quality;
    GLfloatMatrix           vertices;
    IdxMatrix               triangles;
  };

  struct                  Scratch_
  {
    GLfloatVector           vertices;
    GLfloatVector           normals;
  };


 private:

  void                    drawOrbV1_

    ( GLContext&            gtx,
      GLfloat               radius,
      int                   parts = ALL_PARTS );

  void                    drawOrbV1_

    ( GLContext&            gtx,
      GLfloat               radius,
      const Color&          color,
      int                   parts = ALL_PARTS );

  void                    drawOrbV1_

    ( GLContext&            gtx,
      GLfloat               radius,
      const Color&          color1,
      const Color&          color2,
      int                   parts = ALL_PARTS );

  void                    drawOrbV3_

    ( GLContext&            gtx,
      GLfloat               radius,
      int                   parts = ALL_PARTS );

  void                    drawOrbV3_

    ( GLContext&            gtx,
      GLfloat               radius,
      const Color&          color,
      int                   parts = ALL_PARTS );

  void                    drawOrbV3_

    ( GLContext&            gtx,
      GLfloat               radius,
      const Color&          color1,
      const Color&          color2,
      int                   parts = ALL_PARTS );

  static void             initMesh_

    ( Mesh_&                mesh,
      int                   quality );


 private:

  static const int        CACHE_SIZE_ = 6;

  Mesh_                   mcache_ [CACHE_SIZE_ + 1];
  Mesh_*                  current_;
  Scratch_                scratch_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getQuality
//-----------------------------------------------------------------------


inline int Orb::getQuality () const noexcept
{
  return current_->quality;
}


JEM_END_PACKAGE( gl )

#endif
