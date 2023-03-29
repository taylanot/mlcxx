
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

#ifndef JIVE_MODEL_STATEVECTOR_H
#define JIVE_MODEL_STATEVECTOR_H

#include <jive/util/VectorManager.h>
#include <jive/model/import.h>


JIVE_BEGIN_PACKAGE( model )


//-----------------------------------------------------------------------
//   enum StateTag
//-----------------------------------------------------------------------


enum StateTag
{
  STATE0,
  STATE1,
  STATE2
};


extern const StateTag         STATE [3];


//-----------------------------------------------------------------------
//   class StateVector
//-----------------------------------------------------------------------


class StateVector
{
 public:

  typedef util::
    VectorManager             Manager;

  static const char*          NAMES     [3];
  static const char*          OLD_NAMES [3];


  static inline void          store

    ( const Vector&             state,
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline void          store

    ( const Vector&             state,
      StateTag                  tag,
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline void          storeOld

    ( const Vector&             state,
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline void          storeOld

    ( const Vector&             state,
      StateTag                  tag,
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline bool          find

    ( Vector&                   state,
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline bool          find

    ( Vector&                   state,
      StateTag                  tag,
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline bool          findOld

    ( Vector&                   state,
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline bool          findOld

    ( Vector&                   state,
      StateTag                  tag,
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline void          get

    ( Vector&                   state,
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline void          get

    ( Vector&                   state,
      StateTag                  tag,
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline void          getOld

    ( Vector&                   state,
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline void          getOld

    ( Vector&                   state,
      StateTag                  tag,
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline void          updateOld

    ( const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

  static inline void          restoreNew

    ( const Ref<DofSpace>&      dofs,
      const Properties&         globdat );


 private:

  static void                 copy_

    ( const char*               src [3],
      const char*               dest[3],
      const Ref<DofSpace>&      dofs,
      const Properties&         globdat );

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


inline void StateVector::store

  ( const Vector&         state,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Manager::storeVector ( NAMES[0], state, dofs, globdat );
}


inline void StateVector::store

  ( const Vector&         state,
    StateTag              tag,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Manager::storeVector ( NAMES[tag], state, dofs, globdat );
}


//-----------------------------------------------------------------------
//   storeOld
//-----------------------------------------------------------------------


inline void StateVector::storeOld

  ( const Vector&         state,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Manager::storeVector ( OLD_NAMES[0], state, dofs, globdat );
}


inline void StateVector::storeOld

  ( const Vector&         state,
    StateTag              tag,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Manager::storeVector ( OLD_NAMES[tag], state, dofs, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline bool StateVector::find

  ( Vector&               state,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  return Manager::findVector ( state, NAMES[0], dofs, globdat );
}


inline bool StateVector::find

  ( Vector&               state,
    StateTag              tag,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  return Manager::findVector ( state, NAMES[tag], dofs, globdat );
}


//-----------------------------------------------------------------------
//   findOld
//-----------------------------------------------------------------------


inline bool StateVector::findOld

  ( Vector&               state,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  return Manager::findVector ( state, OLD_NAMES[0], dofs, globdat );
}


inline bool StateVector::findOld

  ( Vector&               state,
    StateTag              tag,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  return Manager::findVector ( state, OLD_NAMES[tag], dofs, globdat );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


inline void StateVector::get

  ( Vector&               state,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Manager::getVector ( state, NAMES[0], dofs, globdat );
}


inline void StateVector::get

  ( Vector&               state,
    StateTag              tag,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Manager::getVector ( state, NAMES[tag], dofs, globdat );
}


//-----------------------------------------------------------------------
//   getOld
//-----------------------------------------------------------------------


inline void StateVector::getOld

  ( Vector&               state,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Manager::getVector ( state, OLD_NAMES[0], dofs, globdat );
}


inline void StateVector::getOld

  ( Vector&               state,
    StateTag              tag,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Manager::getVector ( state, OLD_NAMES[tag], dofs, globdat );
}


//-----------------------------------------------------------------------
//   updateOld
//-----------------------------------------------------------------------


inline void StateVector::updateOld

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  copy_ ( NAMES, OLD_NAMES, dofs, globdat );
}


//-----------------------------------------------------------------------
//   restoreNew
//-----------------------------------------------------------------------


inline void StateVector::restoreNew

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  copy_ ( OLD_NAMES, NAMES, dofs, globdat );
}


JIVE_END_PACKAGE( model )

#endif
