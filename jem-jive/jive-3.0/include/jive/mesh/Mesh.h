
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

#ifndef JIVE_MESH_MESH_H
#define JIVE_MESH_MESH_H

#include <jive/fem/XNodeSet.h>
#include <jive/fem/XElementSet.h>
#include <jive/mesh/import.h>


JIVE_BEGIN_PACKAGE( mesh )


//-----------------------------------------------------------------------
//   class Mesh
//-----------------------------------------------------------------------


class Mesh
{
 public:

  typedef Mesh              Self;


  inline                    Mesh      ();

  inline                    Mesh

    ( const ElementSet&       elems );

  inline bool               operator  ==

    ( const Nil&              rhs )          const noexcept;

  inline bool               operator  ==

    ( std::nullptr_t          rhs )          const noexcept;

  inline bool               operator  !=

    ( const Nil&              rhs )          const noexcept;

  inline bool               operator  !=

    ( std::nullptr_t          rhs )          const noexcept;

  inline explicit           operator bool () const noexcept;
  inline bool               operator !    () const noexcept;


 public:

  NodeSet                   nodes;
  ElementSet                elems;

};


//-----------------------------------------------------------------------
//   class XMesh
//-----------------------------------------------------------------------


class XMesh : public Mesh
{
 public:

  typedef XMesh             Self;
  typedef Mesh              Super;


  inline                    XMesh     ();

  inline                    XMesh

    ( const XElementSet&      elems );


 public:

  XNodeSet                  nodes;
  XElementSet               elems;


 private:

  static XNodeSet           getNodes_

    ( const XElementSet&      elems );

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Mesh
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline Mesh::Mesh ()
{}


inline Mesh::Mesh ( const ElementSet& e ) :

  nodes ( e.getNodes() ),
  elems ( e )

{}


//-----------------------------------------------------------------------
//   operator ==
//-----------------------------------------------------------------------


inline bool Mesh::operator == ( const Nil& rhs ) const noexcept
{
  return (elems == rhs);
}


inline bool Mesh::operator == ( std::nullptr_t rhs ) const noexcept
{
  return (elems == rhs);
}


//-----------------------------------------------------------------------
//   operator !=
//-----------------------------------------------------------------------


inline bool Mesh::operator != ( const Nil& rhs ) const noexcept
{
  return (elems != rhs);
}


inline bool Mesh::operator != ( std::nullptr_t rhs ) const noexcept
{
  return (elems != rhs);
}


//-----------------------------------------------------------------------
//   operator bool
//-----------------------------------------------------------------------


inline Mesh::operator bool () const noexcept
{
  return (elems != nullptr);
}


inline bool Mesh::operator ! () const noexcept
{
  return (elems == nullptr);
}


//=======================================================================
//   class XMesh
//=======================================================================


inline XMesh::XMesh ()
{}


inline XMesh::XMesh ( const XElementSet& e ) :

  Super ( e ),

  nodes ( getNodes_( e ) ),
  elems ( e )

{}


JIVE_END_PACKAGE( mesh )

#endif
