#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "png.h"
#include <math.h>
#include <unistd.h>

int width, height;
png_byte color_type;
png_byte bit_depth;
unsigned char* image_data;
png_bytep *row_pointers = NULL;

void read_png_file(char *filename) {
    FILE *fp = fopen(filename, "rb");

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) abort();

    png_infop info = png_create_info_struct(png);
    if(!info) abort();

    if(setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    width      = png_get_image_width(png, info);
    height     = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth  = png_get_bit_depth(png, info);

    png_read_update_info(png, info);

    if (row_pointers) abort();

    int row_bytes = png_get_rowbytes(png, info);
    image_data = (unsigned char*) malloc(row_bytes * height);
    row_pointers = (png_bytep*) malloc(height * sizeof(png_bytep));
    for(int y = 0; y < height; y++) {
        row_pointers[y] = image_data + y * row_bytes;
    }

    png_read_image(png, row_pointers);

    fclose(fp);

//    png_destroy_read_struct(&png, &info, NULL);
}

void write_png_file(char *filename) {
    int y;

    FILE *fp = fopen(filename, "wb");
    if(!fp) abort();

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_GRAY,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);


    if (!row_pointers) abort();

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    fclose(fp);

    png_destroy_write_struct(&png, &info);
}

void write_pgm_file(char* filename) {
    FILE* pgmimg; 
    pgmimg = fopen(filename, "wb"); 
    int i, j, temp;

    fprintf(pgmimg, "P2\n");  
  
    fprintf(pgmimg, "%d %d\n", width, height);  
  
    fprintf(pgmimg, "255\n");  
    int count = 0; 
    for (i = 0; i < height; i++) { 
        for (j = 0; j < width; j++) { 
            temp = image_data[i*width+j]; 
  
            fprintf(pgmimg, "%d ", temp); 
        } 
        fprintf(pgmimg, "\n"); 
    } 
    fclose(pgmimg); 
}

int isvalid(int i, int j) {
    int index = i*width+ j;
    if(index < 0 || index >= width * height)
        return 0;
    return 1;
}

int main(int argc,char *argv[])
{
    int ret, nproc, myid;
    char* filename = argv[1];
    char* histeql = "histeql.pgm";
    char* final = "final.pgm";

    read_png_file(filename);

    int color_depth = 1<<bit_depth;
    int* histogram = (int*)calloc(color_depth, sizeof(int));
    int* histogram_equalized = (int*)calloc(color_depth, sizeof(int));
    long int image_size = width*height;

    unsigned char* image_equalized = (unsigned char*)calloc(image_size, sizeof(unsigned char));
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int rows_per_processor = ((height + nproc - 1)/nproc);
    int bytes_per_processor = rows_per_processor * width;
    int start_row_id = rows_per_processor * myid;
    int end_row_id = start_row_id + rows_per_processor;
    if (end_row_id > image_size) end_row_id = height;

    for(int i = start_row_id; i < end_row_id; i++){
        for(int j = 0; j < width; j++)
            histogram[image_data[i*width + j]]++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, histogram, color_depth, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (myid == 0) {
        int counter = 0;
        for(int i = 0; i < color_depth; i++){
            counter += histogram[i];
            histogram_equalized[i] = (color_depth)*((float)counter/image_size);
        }
    }

    MPI_Bcast(histogram_equalized, color_depth, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    for(int i = start_row_id; i < end_row_id; i++){
        for(int j = 0; j < width; j++)
            image_equalized[i*width + j] = 
                    (unsigned char)histogram_equalized[image_data[i*width + j]];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(image_equalized + myid*bytes_per_processor, bytes_per_processor, MPI_BYTE,
               image_data, bytes_per_processor, MPI_BYTE, 0, MPI_COMM_WORLD);
    
    if (myid == 0) write_pgm_file(histeql);

    float sobelX[3][3] = { { -1, 0, 1 },
                           { -2, 0, 2 },
                           { -1, 0, 1 }
                         };

    float sobelY[3][3] = { { -1, -2, -1 },
                           {  0,  0,  0 },
                           {  1,  2,  1 }
                         };

    //Sobel Computation on submatrices starts here
    unsigned char *sub_sobel =  (unsigned char *)malloc(bytes_per_processor * sizeof(unsigned char));

    int a, b, c, d, e, f, x, y;
    for(int i = start_row_id; i < end_row_id; i++)
    {
        for(int j = 0; j < width; j++)
        {
            a = b = c = d = e = f = 0;
            if(isvalid(i+1, j-1))
                a = image_data[(i+1)*width + j-1];
            if(isvalid(i-1, j-1))
                b = image_data[(i-1)*width + j-1];
            if(isvalid(i+1, j))
                c = image_data[(i+1)*width + j];
            if(isvalid(i-1, j))
                d = image_data[(i-1)*width + j];
            if(isvalid(i+1, j+1))
                e = image_data[(i+1)*width + j+1];
            if(isvalid(i-1, j+1))
                f = image_data[(i-1)*width + j+1];

            x = (a-b) + 2*(c-d) + (e-f);

            a = b = c = d = e = f = 0;
            if(isvalid(i-1, j+1))
                a = image_data[(i-1)*width + j+1];
            if(isvalid(i-1, j-1))
                b = image_data[(i-1)*width + j-1];
            if(isvalid(i,j+1))
                c = image_data[(i)*width + j+1];
            if(isvalid(i,j-1))
                d = image_data[(i)*width + j-1];
            if(isvalid(i+1, j+1))
                e = image_data[(i+1)*width + j+1];
            if(isvalid(i+1, j-1))
                f = image_data[(i+1)*width + j-1];

            y = (a-b) + 2*(c-d) + (e-f);

            sub_sobel[width*(i-start_row_id) + j] = (unsigned char)sqrt(x*x + y*y);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(sub_sobel, bytes_per_processor, MPI_BYTE,
               image_data, bytes_per_processor, MPI_BYTE, 0, MPI_COMM_WORLD);

    if (myid == 0) write_pgm_file(final);

    MPI_Finalize();
}
