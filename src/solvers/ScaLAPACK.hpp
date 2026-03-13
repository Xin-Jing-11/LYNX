#pragma once

// ScaLAPACK and BLACS C declarations for band-parallel subspace diagonalization.
// Only used when USE_SCALAPACK is defined (found by CMake).

#ifdef USE_SCALAPACK

extern "C" {

// ===== BLACS =====
void Cblacs_pinfo(int* mypnum, int* nprocs);
void Cblacs_get(int context, int request, int* value);
void Cblacs_gridinit(int* context, const char* order, int nprow, int npcol);
void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
void Cblacs_gridexit(int context);
void Cblacs_exit(int doneflag);
int  Cblacs_pnum(int context, int prow, int pcol);
void Csys2blacs_handle(int SysCtxt, int* BlacsCtxt);

// ===== ScaLAPACK utility =====
void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
               const int* irsrc, const int* icsrc, const int* ictxt, const int* lld,
               int* info);
int numroc_(const int* n, const int* nb, const int* iproc, const int* isrcproc, const int* nprocs);

// ===== Real ScaLAPACK =====
void pdgemm_(const char* transa, const char* transb,
             const int* m, const int* n, const int* k,
             const double* alpha,
             const double* a, const int* ia, const int* ja, const int* desca,
             const double* b, const int* ib, const int* jb, const int* descb,
             const double* beta,
             double* c, const int* ic, const int* jc, const int* descc);

void pdsyev_(const char* jobz, const char* uplo,
             const int* n, double* a, const int* ia, const int* ja, const int* desca,
             double* w, double* z, const int* iz, const int* jz, const int* descz,
             double* work, const int* lwork, int* info);

void pdsyevd_(const char* jobz, const char* uplo,
              const int* n, double* a, const int* ia, const int* ja, const int* desca,
              double* w, double* z, const int* iz, const int* jz, const int* descz,
              double* work, const int* lwork, int* iwork, const int* liwork, int* info);

void pdpotrf_(const char* uplo, const int* n,
              double* a, const int* ia, const int* ja, const int* desca, int* info);

void pdtrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
             const int* m, const int* n,
             const double* alpha,
             const double* a, const int* ia, const int* ja, const int* desca,
             double* b, const int* ib, const int* jb, const int* descb);

void pdlacpy_(const char* uplo, const int* m, const int* n,
              const double* a, const int* ia, const int* ja, const int* desca,
              double* b, const int* ib, const int* jb, const int* descb);

// ===== Complex ScaLAPACK =====
void pzgemm_(const char* transa, const char* transb,
             const int* m, const int* n, const int* k,
             const void* alpha,
             const void* a, const int* ia, const int* ja, const int* desca,
             const void* b, const int* ib, const int* jb, const int* descb,
             const void* beta,
             void* c, const int* ic, const int* jc, const int* descc);

void pzheev_(const char* jobz, const char* uplo,
             const int* n, void* a, const int* ia, const int* ja, const int* desca,
             double* w, void* z, const int* iz, const int* jz, const int* descz,
             void* work, const int* lwork, double* rwork, const int* lrwork, int* info);

void pzheevd_(const char* jobz, const char* uplo,
              const int* n, void* a, const int* ia, const int* ja, const int* desca,
              double* w, void* z, const int* iz, const int* jz, const int* descz,
              void* work, const int* lwork, double* rwork, const int* lrwork,
              int* iwork, const int* liwork, int* info);

void pzpotrf_(const char* uplo, const int* n,
              void* a, const int* ia, const int* ja, const int* desca, int* info);

void pztrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
             const int* m, const int* n,
             const void* alpha,
             const void* a, const int* ia, const int* ja, const int* desca,
             void* b, const int* ib, const int* jb, const int* descb);

} // extern "C"

#endif // USE_SCALAPACK
