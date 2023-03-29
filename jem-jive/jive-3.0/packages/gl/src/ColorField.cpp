
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


#include <jem/base/assert.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/gl/byte.h>
#include <jem/gl/ColorMap.h>
#include <jem/gl/Palette.h>
#include <jive/gl/Names.h>
#include <jive/gl/utilities.h>
#include <jive/gl/ColorField.h>


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::IllegalInputException;
using jem::gl::toUbyte;
using jem::gl::Color;
using jem::gl::ColorMap;
using jem::util::connect;
using jem::util::disconnect;


//=======================================================================
//   class ColorField
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ColorField::ColorField ( const Ref<Palette>& pal ) :

  palette_ ( pal )

{
  JEM_PRECHECK ( pal );

  minAlpha_  = pal->getMinAlpha ();
  autoScale  = true;
  scaleStep_ = 0.1;

  range_[0]  = range_[1] = 0.0;

  connect ( palette_->changeEvent,
            this, & Self::paletteChanged_ );
}


ColorField::~ColorField ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void ColorField::configure

  ( const Properties&  props,
    const Properties&  dataSets )

{
  if ( props.contains( PropNames::DATA ) )
  {
    dataSource.configure ( PropNames::DATA, props, dataSets );
  }

  props.find ( autoScale,  PropNames::AUTO_SCALE );
  props.find ( scaleStep_, PropNames::SCALE_STEP, 0.0, 1000.0 );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void ColorField::getConfig ( const Properties&  props ) const
{
  dataSource.getConfig ( PropNames::DATA, props );

  props.set ( PropNames::AUTO_SCALE, autoScale  );
  props.set ( PropNames::SCALE_STEP, scaleStep_ );
}


//-----------------------------------------------------------------------
//   getColors
//-----------------------------------------------------------------------


void ColorField::getColors

  ( double&               scale,
    double&               bias,
    const GLubyteMatrix&  colors,
    const Vector&         data,
    const String&         context ) const

{
  JEM_PRECHECK ( colors.size(0) >= 3 &&
                 colors.size(0) <= 4 &&
                 colors.size(1) == data.size() );

  const idx_t  n = data.size ();

  Color        c;


  getScaleAndBias_ ( scale, bias, data, context );

  if ( colors.size(0) == 3_idx )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      palette_->getColor ( c, (bias + scale * data[i]) );

      colors(0,i) = toUbyte ( c.r() );
      colors(1,i) = toUbyte ( c.g() );
      colors(2,i) = toUbyte ( c.b() );
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      palette_->getColor ( c, (bias + scale * data[i]) );

      colors(0,i) = toUbyte ( c.r() );
      colors(1,i) = toUbyte ( c.g() );
      colors(2,i) = toUbyte ( c.b() );
      colors(3,i) = toUbyte ( c.a() );
    }
  }
}


//-----------------------------------------------------------------------
//   getMappedColors
//-----------------------------------------------------------------------


ColorMap* ColorField::getMappedColors

  ( double&               scale,
    double&               bias,
    const GLfloatVector&  colors,
    const Vector&         data,
    const String&         context ) const

{
  JEM_PRECHECK ( colors.size() == data.size() );

  const idx_t  n  = data.size ();

  double       b, c, s;


  getScaleAndBias_ ( scale, bias, data, context );

  if ( ! colorMap_ )
  {
    const_cast<Self*>( this )->initColorMap_ ();
  }

  s =  scale             / (range_[1] - range_[0]);
  b = (bias - range_[0]) / (range_[1] - range_[0]);

  for ( idx_t i = 0; i < n; i++ )
  {
    c = b + s * data[i];

    if      ( c < -1.0e3 )
    {
      c = -1.0e3;
    }
    else if ( c > 1.0e3 )
    {
      c =  1.0e3;
    }

    colors[i] = (GLfloat) c;
  }

  return colorMap_.get ();
}


//-----------------------------------------------------------------------
//   getMinAlpha
//-----------------------------------------------------------------------


GLfloat ColorField::getMinAlpha () const
{
  return minAlpha_;
}


//-----------------------------------------------------------------------
//   setPalette
//-----------------------------------------------------------------------


void ColorField::setPalette ( const Ref<Palette>& pal )
{
  JEM_PRECHECK ( pal );


  disconnect ( palette_->changeEvent,
               this, & Self::paletteChanged_ );

  colorMap_ = nullptr;
  palette_  = pal;
  minAlpha_ = pal->getMinAlpha ();

  range_[0] = range_[1] = 0.0;

  connect ( palette_->changeEvent,
            this, & Self::paletteChanged_ );

}


//-----------------------------------------------------------------------
//   getColorMap
//-----------------------------------------------------------------------


ColorMap* ColorField::getColorMap () const
{
  if ( ! colorMap_ )
  {
    const_cast<Self*>( this )->initColorMap_ ();
  }

  return colorMap_.get ();
}


//-----------------------------------------------------------------------
//   setScaleStep
//-----------------------------------------------------------------------


void ColorField::setScaleStep ( double step )
{
  JEM_PRECHECK ( step >= 0.0 && step < 1000.0 );

  scaleStep_ = step;
}


//-----------------------------------------------------------------------
//   getScaleAndBias_
//-----------------------------------------------------------------------


void ColorField::getScaleAndBias_

  ( double&        scale,
    double&        bias,
    const Vector&  values,
    const String&  context ) const

{
  if ( ! autoScale )
  {
    scale = 1.0;
    bias  = 0.0;
  }
  else
  {
    const double  minVal = palette_->getMinValue ();
    const double  maxVal = palette_->getMaxValue ();

    double        delta  = maxVal - minVal;

    try
    {
      getScaleAndBias ( scale, bias, values, scaleStep_ );
    }
    catch ( IllegalInputException& ex )
    {
      ex.setContext ( context );
      throw;
    }

    if ( delta <= 0.0 )
    {
      delta = 1.0;
    }

    scale *= delta;
    bias   = minVal + delta * bias;
  }
}


//-----------------------------------------------------------------------
//   paletteChanged_
//-----------------------------------------------------------------------


void ColorField::paletteChanged_ ()
{
  colorMap_ = nullptr;
  minAlpha_ = palette_->getMinAlpha ();
  range_[0] = range_[1] = 0.0;
}


//-----------------------------------------------------------------------
//   initColorMap_
//-----------------------------------------------------------------------


void ColorField::initColorMap_ ()
{
  if ( ! colorMap_ )
  {
    colorMap_ = newInstance<ColorMap> ();
  }

  palette_->makeColorMap ( *colorMap_ );

  range_[0] = palette_->getMinValue ();
  range_[1] = palette_->getMaxValue ();

  if ( (range_[1] - range_[0]) <= 0.0 )
  {
    range_[0] = 0.0;
    range_[1] = 1.0;
  }
}


JIVE_END_PACKAGE( gl )
