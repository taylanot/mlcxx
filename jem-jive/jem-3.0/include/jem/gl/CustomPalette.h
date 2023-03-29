
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

#ifndef JEM_GL_CUSTOMPALETTE_H
#define JEM_GL_CUSTOMPALETTE_H

#include <jem/base/Flags.h>
#include <jem/gl/Array.h>
#include <jem/gl/Palette.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class CustomPalette
//-----------------------------------------------------------------------


class CustomPalette : public Palette
{
 public:

  JEM_DECLARE_CLASS       ( CustomPalette, Palette );

  enum                      Option
  {
                              HIDE_MIN_MAX = 1 << 0
  };

  typedef Flags<Option>     Options;


                            CustomPalette ();

                            CustomPalette

    ( const GLfloatMatrix&    colors,
      const Array<double>&    ranges );

  virtual void              getColor

    ( Color&                  col,
      double                  val )          const override;

  virtual double            findValue

    ( double&                 val,
      double                  prec,
      const Color&            col )          const override;

  virtual double            getMinValue   () const override;
  virtual double            getMaxValue   () const override;
  virtual GLfloat           getMinAlpha   () const override;

  virtual void              configure

    ( const Properties&       props )              override;

  virtual void              getConfig

    ( const Properties&       props )        const override;

  virtual void              listProps

    ( const Properties&       props )        const override;

  void                      setColorRanges

    ( const GLfloatMatrix&    colors,
      const Array<double>&    ranges );

  inline GLfloatMatrix      getColors     () const noexcept;
  inline Array<double>      getRanges     () const noexcept;

  void                      setOption

    ( Option                  option,
      bool                    yesno = true );

  void                      setOptions

    ( Options                 options );

  Options                   getOptions    () const noexcept;

  virtual void              fillRectangle

    ( GLContext&              gtx,
      GLint                   width,
      GLint                   height )       const override;

  virtual void              drawTickMarks

    ( GLContext&              gtx,
      TextPainter&            txp,
      GLint                   width,
      GLint                   height,
      GLint                   xtick,
      Alignment               align,
      double                  scale,
      double                  bias )         const override;


 protected:

  virtual                  ~CustomPalette ();


 private:

  static bool               checkRanges_

    ( const Array<double>&    ranges );


 private:

  struct                    Scratch_
  {
    GLfloatVector             buf;
  };

  GLfloatMatrix             colors_;
  Array<double>             ranges_;
  Options                   options_;

  Scratch_                  scratch_;

};


JEM_DEFINE_FLAG_OPS( CustomPalette::Options )


//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getColors
//-----------------------------------------------------------------------


inline GLfloatMatrix CustomPalette::getColors () const noexcept
{
  return colors_;
}


//-----------------------------------------------------------------------
//   getRanges
//-----------------------------------------------------------------------


inline Array<double> CustomPalette::getRanges () const noexcept
{
  return ranges_;
}


JEM_END_PACKAGE( gl )

#endif
