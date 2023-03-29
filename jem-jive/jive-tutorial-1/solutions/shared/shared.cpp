
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/base/Collectable.h>

using namespace jem;


//-----------------------------------------------------------------------
//   class SharedMatrix
//-----------------------------------------------------------------------


class SharedMatrix : public Collectable
{
 protected:

  virtual              ~SharedMatrix ();

 public:

  Array<double,2>       matrix;
};


SharedMatrix::~SharedMatrix ()
{
  System::out() << "SharedMatrix::~SharedMatrix()\n";
}


//-----------------------------------------------------------------------
//   class Builder
//-----------------------------------------------------------------------


class Builder
{
 public:

                        Builder

    ( Ref<SharedMatrix>   mat );

  void                  build ();

 private:

  Ref<SharedMatrix>     matrix_;
};


Builder::Builder ( Ref<SharedMatrix> mat )
{
  matrix_ = mat;
}


void Builder::build ()
{
  matrix_->matrix.resize ( 2, 2 );

  matrix_->matrix(0,0) =  1.0;
  matrix_->matrix(0,1) = -1.0;
  matrix_->matrix(1,0) = -1.0;
  matrix_->matrix(1,1) =  1.0;
}


//-----------------------------------------------------------------------
//   class Solver
//-----------------------------------------------------------------------


class Solver
{
 public:

                        Solver

    ( Ref<SharedMatrix>   mat );

  void                  solve ();

 private:

  Ref<SharedMatrix>     matrix_;
};


Solver::Solver ( Ref<SharedMatrix> mat )
{
  matrix_ = mat;
}


void Solver::solve ()
{
  System::out() << "Matrix:\n\n" << matrix_->matrix << "\n\n";
}


//-----------------------------------------------------------------------
//   test
//-----------------------------------------------------------------------


int test ()
{
  Ref<SharedMatrix>  mat = newInstance<SharedMatrix> ();

  Builder            bld ( mat );
  Solver             sol ( mat );

  bld.build ();
  sol.solve ();

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  System::exec ( test );
}

