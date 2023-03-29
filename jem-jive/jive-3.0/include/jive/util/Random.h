
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

#ifndef JIVE_UTIL_RANDOM_H
#define JIVE_UTIL_RANDOM_H

#include <jem/base/Number.h>
#include <jem/io/Serializable.h>
#include <jem/util/Random.h>
#include <jive/util/import.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class Random
//-----------------------------------------------------------------------


class Random : public jem::Number,
               public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( Random, jem::Number );


  explicit inline         Random

    ( int                   seed = 0 );

  inline void             restart

    ( int                   seed = 0 );

  inline double           next          ();

  inline idx_t            next

    ( idx_t                 ubound );

  inline double           nextGaussian  ();

  virtual String          toString      () const override;
  virtual lint            toInteger     () const override;
  virtual double          toFloat       () const override;
  virtual bool            isFloat       () const override;

  virtual void            readFrom

    ( ObjectInput&          in )                 override;

  virtual void            writeTo

    ( ObjectOutput&         out )          const override;

  static Ref<Random>      get

    ( const Properties&     globdat );

  static void             install

    ( const Properties&     globdat );

  static inline double    getValue

    ( const Properties&     globdat );


 protected:

  virtual                ~Random        ();


 private:

  jem::util::Random       rand_;

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline Random::Random ( int seed ) :

  rand_ ( seed )

{}


//-----------------------------------------------------------------------
//   restart
//-----------------------------------------------------------------------


inline void Random::restart ( int seed )
{
  rand_.restart ( seed );
}


//-----------------------------------------------------------------------
//   next
//-----------------------------------------------------------------------


inline double Random::next ()
{
  return rand_.next ();
}


inline idx_t Random::next ( idx_t ubound )
{
  return rand_.next ( ubound );
}


//-----------------------------------------------------------------------
//   nextGaussian
//-----------------------------------------------------------------------


inline double Random::nextGaussian ()
{
  return rand_.nextGaussian ();
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


inline double Random::getValue ( const Properties& globdat )
{
  return get ( globdat ) -> rand_.next ();
}


JIVE_END_PACKAGE( util )

#endif
