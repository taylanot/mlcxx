
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
#include <jem/base/Once.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/None.h>
#include <jem/util/Flex.h>
#include <jem/util/Properties.h>
#include <jem/util/HashDictionary.h>
#include <jem/gl/LinearPalette.h>
#include <jem/gl/RainbowPalette.h>
#include <jem/gl/CustomPalette.h>
#include <jive/util/CtorMap.h>
#include <jive/util/Factory.h>
#include <jive/gl/GLArray.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/PaletteTable.h>


JEM_DEFINE_CLASS( jive::gl::PaletteTable );


JIVE_BEGIN_PACKAGE( gl )


using jem::staticCast;
using jem::newInstance;
using jem::gl::Palette;
using jem::gl::LinearPalette;
using jem::gl::RainbowPalette;
using jem::gl::CustomPalette;


//=======================================================================
//   Palette constructor funcs
//=======================================================================


static Ref<Palette>  newLinearPalette ()
{
  return newInstance<LinearPalette> ();
}


static Ref<Palette>  newRainbowPalette ()
{
  return newInstance<RainbowPalette> ();
}


static Ref<Palette>  newCustomPalette ()
{
  return newInstance<CustomPalette> ();
}


//=======================================================================
//   class PaletteTable::CtorMap_
//=======================================================================


class PaletteTable::CtorMap_ : public util::CtorMap<PaletteCtor>
{};


//=======================================================================
//   class PaletteTable
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PaletteTable::PaletteTable ()
{
  using jem::gl::Color;

  static jem::Once once = JEM_ONCE_INITIALIZER;

  const int        N    = 5;

  GLfloatMatrix    colors ( 3,  N );
  Vector           ranges ( N + 1 );

  Color            c[N];
  int              i, j;


  jem::runOnce ( once, initOnce_ );

  palettes_ = newInstance<jem::util::HashDict> ();

  // Add the default palettes

  i = 0;

  c[i++] = Color::blue   ();
  c[i++] = Color::green  ();
  c[i++] = Color::yellow ();
  c[i++] = Color::orange ();
  c[i++] = Color::red    ();

  for ( j = 0; j < N; j++ )
  {
    ranges[j] = (double) j / (double) N;

    for ( i = 0; i < 3; i++ )
    {
      colors(i,j) = c[j][i];
    }
  }

  ranges[N] = 1.0;

  palettes_->insert ( "linear",  newInstance<LinearPalette> () );
  palettes_->insert ( "rainbow", newInstance<RainbowPalette>() );

  palettes_->insert (
    "custom",
    newInstance<CustomPalette> ( colors, ranges )
  );
}


PaletteTable::~PaletteTable ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void PaletteTable::configure ( const Properties& props )
{
  using jem::util::isNone;
  using jem::util::Flex;
  using jem::util::DictEnum;

  Ref<DictEnum>  e;
  Ref<Palette>   p;
  Ref<Object>    obj;

  Flex<String>   hitList;
  String         name;


  for ( e = palettes_->enumerate(); ! e->atEnd(); e->toNext() )
  {
    name = e->getKey ();

    if ( ! props.find( obj, name ) )
    {
      continue;
    }

    if ( isNone( obj ) )
    {
      hitList.pushBack ( std::move( name ) );
      continue;
    }

    p = staticCast<Palette> ( e->getValue() );

    p->configure ( props.getProps( name ) );
  }

  if ( hitList.size() )
  {
    e = nullptr;

    for ( idx_t i = 0; i < hitList.size(); i++ )
    {
      palettes_->erase ( hitList[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void PaletteTable::getConfig ( const Properties& props ) const
{
  using jem::util::DictEnum;

  Ref<DictEnum>  e;
  Ref<Palette>   p;

  String         name;


  for ( e = palettes_->enumerate(); ! e->atEnd(); e->toNext() )
  {
    name  = e->getKey ();
    p    = staticCast<Palette> ( e->getValue() );

    p->getConfig ( props.makeProps( name ) );
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void PaletteTable::clear ()
{
  palettes_->clear ();
}


//-----------------------------------------------------------------------
//   addPalette
//-----------------------------------------------------------------------


void PaletteTable::addPalette

  ( const String&        name,
    const Ref<Palette>&  palette )

{
  JEM_PRECHECK ( palette );

  palettes_->insert ( name, palette );
}


//-----------------------------------------------------------------------
//   addPalettes
//-----------------------------------------------------------------------


void PaletteTable::addPalettes

  ( const StringVector&  names,
    const Properties&    conf,
    const Properties&    props )

{
  const idx_t   n = names.size ();

  Ref<Palette>  p;


  for ( idx_t i = 0; i < n; i++ )
  {
    Properties  palConf  = conf .makeProps ( names[i] );
    Properties  palProps = props.getProps  ( names[i] );

    p = newPalette ( palConf, palProps );

    p->configure ( palProps );
    p->getConfig ( palConf  );

    palettes_->insert ( names[i], p );
  }
}


//-----------------------------------------------------------------------
//   findPalette
//-----------------------------------------------------------------------


Ref<Palette> PaletteTable::findPalette

  ( const String& name ) const

{
  return staticCast<Palette> ( palettes_->get( name ) );
}


//-----------------------------------------------------------------------
//   getPalette
//-----------------------------------------------------------------------


Ref<Palette> PaletteTable::getPalette

  ( const String&  name,
    const String&  context ) const

{
  Ref<Palette>  p =

    staticCast<Palette> ( palettes_->get( name ) );

  if ( ! p )
  {
    throw jem::IllegalInputException (
      context,
      String::format ( "undefined palette: `%s\'", name )
    );
  }

  return p;
}


//-----------------------------------------------------------------------
//   listPalettes
//-----------------------------------------------------------------------


StringVector PaletteTable::listPalettes () const
{
  using jem::util::DictEnum;

  Ref<DictEnum>  e;

  StringVector   names ( palettes_->size() );

  idx_t          i     = 0;


  for ( e = palettes_->enumerate(); ! e->atEnd(); e->toNext() )
  {
    names[i++] = e->getKey ();
  }

  return names;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<PaletteTable> PaletteTable::get ( const Properties& viewdat )
{
  Ref<Self>  self;

  viewdat.find ( self, Viewdat::PALETTES );

  if ( ! self )
  {
    self = newInstance<Self> ();

    viewdat.set ( Viewdat::PALETTES, self );
  }

  return self;
}


//-----------------------------------------------------------------------
//   declarePalette
//-----------------------------------------------------------------------


void PaletteTable::declarePalette

  ( const String&  type,
    PaletteCtor    ctor )

{
  JEM_PRECHECK ( ctor );

  CtorMap_::insert ( type, ctor );
}


//-----------------------------------------------------------------------
//   listPaletteTypes
//-----------------------------------------------------------------------


StringVector PaletteTable::listPaletteTypes ()
{
  return CtorMap_::listKnownTypes ();
}


//-----------------------------------------------------------------------
//   newPalette
//-----------------------------------------------------------------------


Ref<Palette> PaletteTable::newPalette

  ( const String& type )

{
  PaletteCtor ctor = CtorMap_::find ( type );

  if ( ctor )
  {
    return ctor ();
  }
  else
  {
    return nullptr;
  }
}


Ref<Palette> PaletteTable::newPalette

  ( const Properties&  conf,
    const Properties&  props )

{
  using jive::util::Factory;

  Ref<Palette>  palette;
  String        type;

  props.get ( type, Factory::TYPE_PROP );
  conf .set ( Factory::TYPE_PROP, type );

  palette = newPalette ( type );

  if ( ! palette )
  {
    Factory::noSuchTypeError (
      props.getContext ( Factory::TYPE_PROP ),
      type,
      "palette",
      listPaletteTypes ()
    );
  }

  return palette;
}


//-----------------------------------------------------------------------
//   initOnce_
//-----------------------------------------------------------------------


void PaletteTable::initOnce_ ()
{
  CtorMap_::insert ( "Linear",  & newLinearPalette  );
  CtorMap_::insert ( "Rainbow", & newRainbowPalette );
  CtorMap_::insert ( "Custom",  & newCustomPalette  );
}


JIVE_END_PACKAGE( gl )
