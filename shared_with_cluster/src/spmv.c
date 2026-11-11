
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>
#include <windows.h>

typedef struct { int rows, cols; long long nnz; int *r,*c; double *v; } COO;
typedef struct { int rows, cols; long long nnz; int *rowptr,*col; double *val; } CSR;
typedef struct { int rows, cols, width; long long nnz; int *col; double *val; } ELL;

static double now_ms(void){
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;

    if(frequency.QuadPart == 0){
        if(!QueryPerformanceFrequency(&frequency)){
            fprintf(stderr, "ERROR: QueryPerformanceFrequency failed\n");
            exit(2);
        }
    }

    if(!QueryPerformanceCounter(&counter)){
        fprintf(stderr, "ERROR: QueryPerformanceCounter failed\n");
        exit(2);
    }

    return (double)counter.QuadPart * 1000.0 /
           (double)frequency.QuadPart;
}
static void die(const char *s){ fprintf(stderr,"ERROR: %s\n",s); exit(2); }
static char *skip_line(char *p){ while(*p && *p!='\n') p++; return *p?p+1:p; }

static COO read_mtx(const char *path){
    FILE *f=fopen(path,"r");
    if(!f) die("cannot open Matrix Market file");

    char line[4096], banner[256], object[64], format[64], field[64], symmetry[64];
    if(!fgets(line,sizeof(line),f)) die("empty matrix file");
    if(sscanf(line,"%%%%MatrixMarket %63s %63s %63s %63s",object,format,field,symmetry)!=4)
        die("not a Matrix Market file");
    if(strcmp(object,"matrix") || strcmp(format,"coordinate"))
        die("only Matrix Market coordinate matrices are supported");

    int symmetric=(!strcmp(symmetry,"symmetric") || !strcmp(symmetry,"hermitian"));
    int pattern=!strcmp(field,"pattern");
    int complex_field=!strcmp(field,"complex");
    if(complex_field) die("complex Matrix Market values are not supported; choose real/integer/pattern");

    do {
        if(!fgets(line,sizeof(line),f)) die("missing Matrix Market dimensions");
    } while(line[0]=='%' || line[0]=='\n' || line[0]=='\r');

    int rows,cols;
    long long nz;
    if(sscanf(line,"%d %d %lld",&rows,&cols,&nz)!=3 || rows<=0 || cols<=0 || nz<0)
        die("invalid Matrix Market dimensions");

    long long cap=symmetric?2*nz:nz;
    COO a={0};
    a.rows=rows;
    a.cols=cols;
    a.r=(int*)malloc((size_t)cap*sizeof(int));
    a.c=(int*)malloc((size_t)cap*sizeof(int));
    a.v=(double*)malloc((size_t)cap*sizeof(double));
    if(!a.r||!a.c||!a.v) die("matrix allocation failed");

    long long k=0;
    while(k<nz && fgets(line,sizeof(line),f)){
        if(line[0]=='%' || line[0]=='\n' || line[0]=='\r') continue;

        int r,c;
        double v=1.0;
        int n=pattern?sscanf(line,"%d %d",&r,&c):sscanf(line,"%d %d %lf",&r,&c,&v);
        if(n<(pattern?2:3)) continue;

        r--;
        c--;
        if(r<0||r>=rows||c<0||c>=cols) die("matrix index out of range");

        a.r[a.nnz]=r;
        a.c[a.nnz]=c;
        a.v[a.nnz]=v;
        a.nnz++;

        if(symmetric && r!=c){
            a.r[a.nnz]=c;
            a.c[a.nnz]=r;
            a.v[a.nnz]=v;
            a.nnz++;
        }
        k++;
    }

    fclose(f);
    if(k!=nz) die("Matrix Market file ended before declared NNZ");
    return a;
}

static void free_coo(COO*a){free(a->r);free(a->c);free(a->v);}

static CSR to_csr(const COO*a){
    CSR b={0};
    b.rows=a->rows;
    b.cols=a->cols;
    b.nnz=a->nnz;

    b.rowptr=(int*)calloc((size_t)b.rows+1,sizeof(int));
    b.col=(int*)malloc((size_t)b.nnz*sizeof(int));
    b.val=(double*)malloc((size_t)b.nnz*sizeof(double));
    if(!b.rowptr||!b.col||!b.val) die("CSR allocation failed");

    for(long long k=0;k<a->nnz;k++) b.rowptr[a->r[k]+1]++;
    for(int i=0;i<b.rows;i++) b.rowptr[i+1]+=b.rowptr[i];

    int *next=(int*)malloc((size_t)b.rows*sizeof(int));
    if(!next) die("allocation failed");
    memcpy(next,b.rowptr,(size_t)b.rows*sizeof(int));

    for(long long k=0;k<a->nnz;k++){
        int p=next[a->r[k]]++;
        b.col[p]=a->c[k];
        b.val[p]=a->v[k];
    }

    free(next);
    return b;
}

static ELL to_ell(const CSR*a){
    ELL e={0};
    e.rows=a->rows;
    e.cols=a->cols;
    e.nnz=a->nnz;

    for(int i=0;i<a->rows;i++){
        int n=a->rowptr[i+1]-a->rowptr[i];
        if(n>e.width) e.width=n;
    }

    if(e.width==0) e.width=1;

    size_t count=(size_t)e.rows*(size_t)e.width;
    if(count>SIZE_MAX/sizeof(double) || count>SIZE_MAX/sizeof(int))
        die("ELL size overflow");

    e.col=(int*)malloc(count*sizeof(int));
    e.val=(double*)calloc(count,sizeof(double));
    if(!e.col||!e.val) die("ELL allocation failed");

    for(size_t p=0;p<count;p++) e.col[p]=-1;

    for(int i=0;i<a->rows;i++){
        int q=0;
        for(int k=a->rowptr[i];k<a->rowptr[i+1];k++,q++){
            size_t p=(size_t)i*e.width+q;
            e.col[p]=a->col[k];
            e.val[p]=a->val[k];
        }
    }
    return e;
}

static void free_csr(CSR*a){free(a->rowptr);free(a->col);free(a->val);}
static void free_ell(ELL*a){free(a->col);free(a->val);}

static void spmv_coo(const COO*a,const double*x,double*y,int omp,int threads){
    memset(y,0,(size_t)a->rows*sizeof(double));

    if(omp){
        #pragma omp parallel for num_threads(threads) schedule(runtime)
        for(long long k=0;k<a->nnz;k++){
            double z=a->v[k]*x[a->c[k]];
            #pragma omp atomic update
            y[a->r[k]] += z;
        }
    } else {
        for(long long k=0;k<a->nnz;k++)
            y[a->r[k]]+=a->v[k]*x[a->c[k]];
    }
}

static void spmv_csr(const CSR*a,const double*x,double*y,int omp,int threads){
    if(omp){
        #pragma omp parallel for num_threads(threads) schedule(runtime)
        for(int i=0;i<a->rows;i++){
            double s=0;
            for(int k=a->rowptr[i];k<a->rowptr[i+1];k++)
                s+=a->val[k]*x[a->col[k]];
            y[i]=s;
        }
    } else {
        for(int i=0;i<a->rows;i++){
            double s=0;
            for(int k=a->rowptr[i];k<a->rowptr[i+1];k++)
                s+=a->val[k]*x[a->col[k]];
            y[i]=s;
        }
    }
}

static void spmv_ell(const ELL*a,const double*x,double*y,int omp,int threads){
    if(omp){
        #pragma omp parallel for num_threads(threads) schedule(runtime)
        for(int i=0;i<a->rows;i++){
            double s=0;
            size_t base=(size_t)i*a->width;
            for(int j=0;j<a->width;j++){
                int c=a->col[base+j];
                if(c>=0) s+=a->val[base+j]*x[c];
            }
            y[i]=s;
        }
    } else {
        for(int i=0;i<a->rows;i++){
            double s=0;
            size_t base=(size_t)i*a->width;
            for(int j=0;j<a->width;j++){
                int c=a->col[base+j];
                if(c>=0) s+=a->val[base+j]*x[c];
            }
            y[i]=s;
        }
    }
}

static double maxerr(const double*a,const double*b,int n){
    double m=0;
    for(int i=0;i<n;i++){
        double d=fabs(a[i]-b[i]);
        if(d>m) m=d;
    }
    return m;
}

static void read_vector(const char*path,double*x,int n){
    FILE*f=fopen(path,"r");
    if(!f) die("cannot open vector file");
    for(int i=0;i<n;i++)
        if(fscanf(f,"%lf",&x[i])!=1) die("vector file has too few values");
    fclose(f);
}

static void write_vector(const char*path,const double*y,int n){
    FILE*f=fopen(path,"w");
    if(!f) die("cannot write output vector");
    for(int i=0;i<n;i++) fprintf(f,"%.17g\n",y[i]);
    fclose(f);
}

int main(int argc,char**argv){
    const char *matrix=NULL,*vector=NULL,*dataset=NULL;
    const char *format="csr",*mode="seq",*sched="static";
    const char *raw=NULL,*out=NULL,*refout=NULL;
    int threads=1,chunk=1,runs=6,repetitions=100;

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--matrix")&&i+1<argc) matrix=argv[++i];
        else if(!strcmp(argv[i],"--vector")&&i+1<argc) vector=argv[++i];
        else if(!strcmp(argv[i],"--dataset")&&i+1<argc) dataset=argv[++i];
        else if(!strcmp(argv[i],"--format")&&i+1<argc) format=argv[++i];
        else if(!strcmp(argv[i],"--mode")&&i+1<argc) mode=argv[++i];
        else if(!strcmp(argv[i],"--schedule")&&i+1<argc) sched=argv[++i];
        else if(!strcmp(argv[i],"--threads")&&i+1<argc) threads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--chunk")&&i+1<argc) chunk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--runs")&&i+1<argc) runs=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--repetitions")&&i+1<argc) repetitions=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--raw")&&i+1<argc) raw=argv[++i];
        else if(!strcmp(argv[i],"--output-vector")&&i+1<argc) out=argv[++i];
        else if(!strcmp(argv[i],"--reference-vector")&&i+1<argc) refout=argv[++i];
        else die("unknown/incomplete argument");
    }

    if(!matrix||!vector||!dataset||threads<1||runs<2||repetitions<1)
        die("usage: --matrix FILE --vector FILE --dataset ID [--format coo|csr|ell] [--mode seq|omp] [--schedule static|dynamic|guided] [--threads N] [--chunk N] [--runs N] [--raw FILE]");

    omp_sched_t os=!strcmp(sched,"dynamic")?omp_sched_dynamic:
                   (!strcmp(sched,"guided")?omp_sched_guided:omp_sched_static);
    omp_set_schedule(os,chunk);

   COO a=read_mtx(matrix);
CSR c=to_csr(&a);
ELL e={0};

if(!strcmp(format,"ell")){
    e=to_ell(&c);
}
    double*x=(double*)malloc((size_t)a.cols*sizeof(double));
    double*y=(double*)malloc((size_t)a.rows*sizeof(double));
    double*ref=(double*)malloc((size_t)a.rows*sizeof(double));

    if(!x||!y||!ref) die("vector allocation failed");
    read_vector(vector,x,a.cols);

    spmv_csr(&c,x,ref,0,1);
    if(refout) write_vector(refout,ref,a.rows);

    double err=0;
    volatile double benchmark_sink = 0.0;
    if(!strcmp(format,"coo")){
        spmv_coo(&a,x,y,0,1);
        err=maxerr(ref,y,a.rows);
    } else if(!strcmp(format,"csr")){
        spmv_csr(&c,x,y,0,1);
        err=maxerr(ref,y,a.rows);
    } else if(!strcmp(format,"ell")){
        spmv_ell(&e,x,y,0,1);
        err=maxerr(ref,y,a.rows);
    } else {
        die("format must be coo, csr or ell");
    }

    double refmax=0.0;
    for(int qi=0;qi<a.rows;qi++)
        if(fabs(ref[qi])>refmax) refmax=fabs(ref[qi]);

    if(err>1e-10*(1.0+refmax)){
        fprintf(stderr,"ERROR: sequential format differs from CSR reference; max_abs_error=%.17g\n",err);
        die("correctness check failed");
    }

    FILE*f=NULL;
    if(raw){
        f=fopen(raw,"a");
        if(!f) die("cannot open raw CSV");
    }

    for(int run=0;run<runs;run++){
        double t0=now_ms();

        for(int rep=0;rep<repetitions;rep++){
            if(!strcmp(format,"coo"))
                spmv_coo(&a,x,y,!strcmp(mode,"omp"),threads);
            else if(!strcmp(format,"csr"))
                spmv_csr(&c,x,y,!strcmp(mode,"omp"),threads);
            else
                spmv_ell(&e,x,y,!strcmp(mode,"omp"),threads);

            /* Keep each result observable so repeated calls are not removed. */
            benchmark_sink += y[rep % a.rows];
        }

        double total_ms=now_ms()-t0;
        double ms=total_ms/repetitions;
        err=maxerr(ref,y,a.rows);

        if(f){
            const char *pb=getenv("OMP_PROC_BIND");
            fprintf(f,"ClusterB,P5,CPU,%s_%s,%s,%lld,%d,%d,%.9f,%.9f,,%.9f,%.17g,,%d,%s,%d,,%s\n",
                format,mode,dataset,a.nnz,threads,run,total_ms,ms,
                (2.0*(double)a.nnz)/(ms*1e6),err,run==0,sched,chunk,pb?pb:"");
        }

        printf("run=%d warmup=%d repetitions=%d total_ms=%.6f avg_ms=%.9f max_abs_error=%.12g GFLOP/s=%.6f\n",
            run,run==0,repetitions,total_ms,ms,err,
            (2.0*(double)a.nnz)/(ms*1e6));
    }

    printf("benchmark_sink=%g\n", (double)benchmark_sink);

    if(out) write_vector(out,y,a.rows);
    if(f) fclose(f);
    if(!strcmp(format,"ell")){
    printf("rows=%d cols=%d nnz=%lld ell_width=%d ell_padding_ratio=%.6f\n",
        a.rows,a.cols,a.nnz,e.width,
        ((double)a.rows*e.width)/(a.nnz?a.nnz:1));
} else {
    printf("rows=%d cols=%d nnz=%lld\n",a.rows,a.cols,a.nnz);
}

   
    free(x);
    free(y);
    free(ref);
    free_coo(&a);
    free_csr(&c);
    free_ell(&e);
    return 0;
}
