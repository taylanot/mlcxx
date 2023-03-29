
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

#ifndef JIVE_GL_COLORFIELD_H
#define JIVE_GL_COLORFIELD_H

#include <jive/gl/import.h>
#include <jive/gl/GLArray.h>
#include <jive/gl/DataSource.h>


namespace jem
{
  namespace gl
  {
    class Palette;
  }
}


JIVE_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class ColorField
//-----------------------------------------------------------------------


class ColorField : public Object
{
 public:

  typedef ColorField        Self;
  typedef Object            Super;
  typedef jem::
    gl::Palette             Palette;


  explicit                  ColorField

    ( const Ref<Palette>&     pal );

  void                      configure

    ( const Properties&       props,
      const Properties&       dataSets );

  void                      getConfig

    ( const Properties&       props )          const;

  void                      getColors

    ( double&                 scale,
      double&                 bias,
      const GLubyteMatrix&    colors,
      const Vector&           data,
      const String&           context )        const;

  ColorMap*                 getMappedColors

    ( double&                 scale,
      double&                 bias,
      const GLfloatVector&    colors,
      const Vector&           data,
      const String&           context )        const;

  GLfloat                   getMinAlpha     () const;
  inline Palette*           getPalette      () const;

  void                      setPalette

    ( const Ref<Palette>&     pal );

  ColorMap*                 getColorMap     () const;
  inline double             getScaleStep    () const;

  void                      setScaleStep

    ( double                  step );


 public:

  DataSource                dataSource;
  bool                      autoScale;


 protected:

                           ~ColorField      ();


 private:

  void                      getScaleAndBias_

    ( double&                 scale,
      double&                 bias,
      const Vector&           values,
      const String&           context )        const;

  void                      paletteChanged_ ();
  void                      initColorMap_   ();


 private:

  Ref<Palette>              palette_;
  Ref<ColorMap>             colorMap_;
  GLfloat                   minAlpha_;
  double                    scaleStep_;
  double                    range_[2];

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getPalette
//-----------------------------------------------------------------------


inline jem::gl::Palette* ColorField::getPalette () const
{
  return palette_.get ();
}


//-----------------------------------------------------------------------
//   getScaleStep
//-----------------------------------------------------------------------


inline double ColorField::getScaleStep () const
{
  return scaleStep_;
}


JIVE_END_PACKAGE( gl )

#endif
