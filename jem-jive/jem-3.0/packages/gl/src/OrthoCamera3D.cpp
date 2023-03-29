
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


#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/OrthoCamera3D.h>


JEM_DEFINE_CLASS( jem::gl::OrthoCamera3D );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class OrthoCamera3D
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


OrthoCamera3D::OrthoCamera3D

  ( const String&        name,
    const Ref<GfxNode>&  child ) :

    Super ( name, child )

{}


OrthoCamera3D::~OrthoCamera3D ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void OrthoCamera3D::configure ( const Properties& props )
{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    Vertex3d    pos;
    double      r;


    if ( myProps.find( r, PropNames::RANGE, 1.0e-20, 1.0e20 ) )
    {
      range_    = (GLdouble) r;
      nearClip_ = -range_;
      farClip_  =  range_;
      newProj_  =  true;
    }

    pos = 0.0;

    if ( Vertex::configure( pos, PropNames::POSITION, myProps ) )
    {
      transfm_.setTranslation ( -pos[0], -pos[1], -pos[2] );
    }
  }

  Super::configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void OrthoCamera3D::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  Vertex3d    pos;

  pos[0] = -transfm_(0,3);
  pos[1] = -transfm_(1,3);
  pos[2] = -transfm_(2,3);

  myProps.set ( PropNames::RANGE, range_ );

  Vertex::getConfig ( PropNames::POSITION, myProps, pos );
  Super ::getConfig ( props );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void OrthoCamera3D::listProps ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::RANGE,
                "The cross-screen visible distance" );
  myProps.set ( PropNames::POSITION,
                "The position of the camera" );

  Super::listProps ( props );
}


JEM_END_PACKAGE( gl )
