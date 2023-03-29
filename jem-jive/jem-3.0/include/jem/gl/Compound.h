
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

#ifndef JEM_GL_COMPOUND_H
#define JEM_GL_COMPOUND_H

#include <jem/base/Interface.h>
#include <jem/gl/GfxNode.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class Compound
//-----------------------------------------------------------------------


class Compound : public Interface
{
 public:

  virtual idx_t             childCount    () const = 0;

  virtual void              addChild

    ( const Ref<GfxNode>&     child )              = 0;

  virtual bool              eraseChild

    ( const Ref<GfxNode>&     child )              = 0;


 protected:

  virtual inline           ~Compound      ();


 private:

  virtual void              emitVtable_   ();

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline bool                 isCompound

  ( const Ref<GfxNode>&       node );

inline Compound*            toCompound

  ( const Ref<GfxNode>&       node );

Compound&                   toCompound

  ( GfxNode&                  node );

inline void                 addChild

  ( GfxNode&                  parent,
    const Ref<GfxNode>&       child );

inline void                 eraseChild

  ( GfxNode&                  parent,
    const Ref<GfxNode>&       child );

bool                        addToScene

  ( const Ref<GfxNode>&       scene,
    const Ref<GfxNode>&       node );

void                        eraseFromParent

  ( const Ref<GfxNode>&       node );



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


inline Compound::~Compound ()
{}


//-----------------------------------------------------------------------
//   isCompound
//-----------------------------------------------------------------------


inline bool             isCompound

  ( const Ref<GfxNode>&   node )

{
  return (dynamicCast<Compound*>( node ) != nullptr);
}


//-----------------------------------------------------------------------
//   toCompound
//-----------------------------------------------------------------------


inline Compound*        toCompound

  ( const Ref<GfxNode>&   node )

{
  return dynamicCast<Compound*> ( node );
}


//-----------------------------------------------------------------------
//   addChild
//-----------------------------------------------------------------------


inline void             addChild

  ( GfxNode&              parent,
    const Ref<GfxNode>&   child )

{
  toCompound(parent).addChild ( child );
}


//-----------------------------------------------------------------------
//   eraseChild
//-----------------------------------------------------------------------


inline void             eraseChild

  ( GfxNode&              parent,
    const Ref<GfxNode>&   child )

{
  toCompound(parent).eraseChild ( child );
}


JEM_END_PACKAGE( gl )

#endif
