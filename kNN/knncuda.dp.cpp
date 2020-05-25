#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <mkl_blas_sycl.hpp>
#include <mkl_lapack_sycl.hpp>
#include <mkl_sycl_types.hpp>
#include <dpct/blas_utils.hpp>

#define BLOCK_DIM 16


/**
 * Computes the squared Euclidean distance matrix between the query points and the reference points.
 *
 * @param ref          refence points stored in the global memory
 * @param ref_width    number of reference points
 * @param ref_pitch    pitch of the reference points array in number of column
 * @param query        query points stored in the global memory
 * @param query_width  number of query points
 * @param query_pitch  pitch of the query points array in number of columns
 * @param height       dimension of points = height of texture `ref` and of the array `query`
 * @param dist         array containing the query_width x ref_width computed distances
 */
void compute_distances(float * ref,
                                  int     ref_width,
                                  int     ref_pitch,
                                  float * query,
                                  int     query_width,
                                  int     query_pitch,
                                  int     height,
                                  float * dist,
                                  sycl::nd_item<3> item_ct1,
                                  dpct::accessor<float, dpct::local, 2> shared_A,
                                  dpct::accessor<float, dpct::local, 2> shared_B,
                                  int *begin_A,
                                  int *begin_B,
                                  int *step_A,
                                  int *step_B,
                                  int *end_A) {

    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B

    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)

    // Thread index
    int tx = item_ct1.get_local_id(2);
    int ty = item_ct1.get_local_id(1);

    // Initializarion of the SSD for the current thread
    float ssd = 0.f;

    // Loop parameters
    *begin_A = BLOCK_DIM * item_ct1.get_group(1);
    *begin_B = BLOCK_DIM * item_ct1.get_group(2);
    *step_A = BLOCK_DIM * ref_pitch;
    *step_B = BLOCK_DIM * query_pitch;
    *end_A = *begin_A + (height - 1) * ref_pitch;

    // Conditions
    int cond0 = (*begin_A + tx < ref_width); // used to write in shared memory
    int cond1 = (*begin_B + tx <
                 query_width); // used to write in shared memory & to
                               // computations and to write in output array
    int cond2 =
        (*begin_A + ty <
         ref_width); // used to computations and to write in output matrix

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = (*begin_A), b = (*begin_B); a <= *end_A;
         a += *step_A, b += *step_B) {

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a/ref_pitch + ty < height) {
            shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? query[b + query_pitch * ty + tx] : 0;
        }
        else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        item_ct1.barrier();

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k){
                float tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp*tmp;
            }
        }

        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
        item_ct1.barrier();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1) {
        dist[(*begin_A + ty) * query_pitch + *begin_B + tx] = ssd;
    }
}


/**
 * Computes the squared Euclidean distance matrix between the query points and the reference points.
 *
 * @param ref          refence points stored in the texture memory
 * @param ref_width    number of reference points
 * @param query        query points stored in the global memory
 * @param query_width  number of query points
 * @param query_pitch  pitch of the query points array in number of columns
 * @param height       dimension of points = height of texture `ref` and of the array `query`
 * @param dist         array containing the query_width x ref_width computed distances
 */
void compute_distance_texture(dpct::image_accessor<float, 2> ref, int ref_width,
                              float *query, int query_width, int query_pitch,
                              int height, float *dist,
                              sycl::nd_item<3> item_ct1) {
    unsigned int xIndex =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
        item_ct1.get_local_id(2);
    unsigned int yIndex =
        item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
        item_ct1.get_local_id(1);
    if ( xIndex<query_width && yIndex<ref_width) {
        float ssd = 0.f;
        for (int i=0; i<height; i++) {
            float tmp = dpct::read_image(ref, (float)yIndex, (float)i) -
                        query[i * query_pitch + xIndex];
            ssd += tmp * tmp;
        }
        dist[yIndex * query_pitch + xIndex] = ssd;
    }
}


/**
 * For each reference point (i.e. each column) finds the k-th smallest distances
 * of the distance matrix and their respective indexes and gathers them at the top
 * of the 2 arrays.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 *
 * @param dist         distance matrix
 * @param dist_pitch   pitch of the distance matrix given in number of columns
 * @param index        index matrix
 * @param index_pitch  pitch of the index matrix given in number of columns
 * @param width        width of the distance matrix and of the index matrix
 * @param height       height of the distance matrix
 * @param k            number of values to find
 */
void modified_insertion_sort(float * dist,
                                        int     dist_pitch,
                                        int *   index,
                                        int     index_pitch,
                                        int     width,
                                        int     height,
                                        int     k,
                                        sycl::nd_item<3> item_ct1){

    // Column position
    unsigned int xIndex =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
        item_ct1.get_local_id(2);

    // Do nothing if we are out of bounds
    if (xIndex < width) {

        // Pointer shift
        float * p_dist  = dist  + xIndex;
        int *   p_index = index + xIndex;

        // Initialise the first index
        p_index[0] = 0;

        // Go through all points
        for (int i=1; i<height; ++i) {

            // Store current distance and associated index
            float curr_dist = p_dist[i*dist_pitch];
            int   curr_index  = i;

            // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
            if (i >= k && curr_dist >= p_dist[(k-1)*dist_pitch]) {
                continue;
            }

            // Shift values (and indexes) higher that the current distance to the right
            int j = sycl::min(i, (int)(k - 1));
            while (j > 0 && p_dist[(j-1)*dist_pitch] > curr_dist) {
                p_dist[j*dist_pitch]   = p_dist[(j-1)*dist_pitch];
                p_index[j*index_pitch] = p_index[(j-1)*index_pitch];
                --j;
            }

            // Write the current distance and index at their position
            p_dist[j*dist_pitch]   = curr_dist;
            p_index[j*index_pitch] = curr_index; 
        }
    }
}


/**
 * Computes the square root of the first k lines of the distance matrix.
 *
 * @param dist   distance matrix
 * @param width  width of the distance matrix
 * @param pitch  pitch of the distance matrix given in number of columns
 * @param k      number of values to consider
 */
void compute_sqrt(float * dist, int width, int pitch, int k,
                  sycl::nd_item<3> item_ct1){
    unsigned int xIndex =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
        item_ct1.get_local_id(2);
    unsigned int yIndex =
        item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
        item_ct1.get_local_id(1);
    if (xIndex<width && yIndex<k)
        dist[yIndex * pitch + xIndex] =
            sycl::sqrt((double)(dist[yIndex * pitch + xIndex]));
}


/**
 * Computes the squared norm of each column of the input array.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param height  number of rows of `array` = dimension of the points
 * @param norm    output array containing the squared norm values
 */
void compute_squared_norm(float * array, int width, int pitch, int height, float * norm,
                          sycl::nd_item<3> item_ct1){
    unsigned int xIndex =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
        item_ct1.get_local_id(2);
    if (xIndex<width){
        float sum = 0.f;
        for (int i=0; i<height; i++){
            float val = array[i*pitch+xIndex];
            sum += val*val;
        }
        norm[xIndex] = sum;
    }
}


/**
 * Add the reference points norm (column vector) to each colum of the input array.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param height  number of rows of `array` = dimension of the points
 * @param norm    reference points norm stored as a column vector
 */
void add_reference_points_norm(float * array, int width, int pitch, int height, float * norm,
                               sycl::nd_item<3> item_ct1,
                               float *shared_vec){
    unsigned int tx = item_ct1.get_local_id(2);
    unsigned int ty = item_ct1.get_local_id(1);
    unsigned int xIndex =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + tx;
    unsigned int yIndex =
        item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + ty;

    if (tx==0 && yIndex<height)
        shared_vec[ty] = norm[yIndex];
    item_ct1.barrier();
    if (xIndex<width && yIndex<height)
        array[yIndex*pitch+xIndex] += shared_vec[ty];
}


/**
 * Adds the query points norm (row vector) to the k first lines of the input
 * array and computes the square root of the resulting values.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param k       number of neighbors to consider
 * @param norm     query points norm stored as a row vector
 */
void add_query_points_norm_and_sqrt(float * array, int width, int pitch, int k, float * norm,
                                    sycl::nd_item<3> item_ct1){
    unsigned int xIndex =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
        item_ct1.get_local_id(2);
    unsigned int yIndex =
        item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
        item_ct1.get_local_id(1);
    if (xIndex<width && yIndex<k)
        array[yIndex * pitch + xIndex] =
            sycl::sqrt((double)(array[yIndex * pitch + xIndex] + norm[xIndex]));
}

bool knn_cuda_global(const float *ref, int ref_nb, const float *query,
                     int query_nb, int dim, int k, float *knn_dist,
                     int *knn_index) try {

    // Constants
    const unsigned int size_of_float = sizeof(float);
    const unsigned int size_of_int   = sizeof(int);

    // Return variables
    int err0, err1, err2, err3;

    // Check that we have at least one CUDA device 
    int nb_devices;
    /*
    DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    err0 = (nb_devices = dpct::dev_mgr::instance().device_count(), 0);
    if (err0 != 0 || nb_devices == 0) {
        printf("ERROR: No CUDA device found\n");
        return false;
    }

    // Select the first CUDA device as default
    /*
    DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    err0 = (dpct::dev_mgr::instance().select_device(0), 0);

    // Allocate global memory
    float * ref_dev   = NULL;
    float * query_dev = NULL;
    float * dist_dev  = NULL;
    int   * index_dev = NULL;
    size_t  ref_pitch_in_bytes;
    size_t  query_pitch_in_bytes;
    size_t  dist_pitch_in_bytes;
    size_t  index_pitch_in_bytes;
    /*
    DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    dpct::dpct_malloc((void **)&ref_dev, &ref_pitch_in_bytes,
        ref_nb * size_of_float, dim);
    printf("ref_dev: %p ", ref_dev);
    /*
    DPCT1003:3: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    err1 = (dpct::dpct_malloc((void **)&query_dev, &query_pitch_in_bytes,
                              query_nb * size_of_float, dim),
            0);
    printf("query_dev: %p ", query_dev);
    /*
    DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    err2 = (dpct::dpct_malloc((void **)&dist_dev, &dist_pitch_in_bytes,
                              query_nb * size_of_float, ref_nb),
            0);
    printf("dist_dev: %p ", dist_dev);
    /*
    DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    err3 = (dpct::dpct_malloc((void **)&index_dev, &index_pitch_in_bytes,
                              query_nb * size_of_int, k),
            0);
    printf("index_dev: %p\n\n", index_dev);
    if (err0 != 0 || err1 != 0 || err2 != 0 || err3 != 0) {
        printf("ERROR: Memory allocation error\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        return false;
    }

    // Deduce pitch values
    size_t ref_pitch   = ref_pitch_in_bytes   / size_of_float;
    size_t query_pitch = query_pitch_in_bytes / size_of_float;
    size_t dist_pitch  = dist_pitch_in_bytes  / size_of_float;
    size_t index_pitch = index_pitch_in_bytes / size_of_int;

    // Check pitch values
    if (query_pitch != dist_pitch || query_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        return false; 
    }

    // Copy reference and query data from the host to the device
    /*
    DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    err0 = (dpct::dpct_memcpy(ref_dev, ref_pitch_in_bytes, ref,
                              ref_nb * size_of_float, ref_nb * size_of_float,
                              dim, dpct::host_to_device),
            0);
    /*
    DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    err1 =
        (dpct::dpct_memcpy(query_dev, query_pitch_in_bytes, query,
                           query_nb * size_of_float, query_nb * size_of_float,
                           dim, dpct::host_to_device),
         0);
    if (err0 != 0 || err1 != 0) {
        printf("ERROR: Unable to copy data from host to device\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        return false; 
    }

    // Compute the squared Euclidean distances
    sycl::range<3> block0(BLOCK_DIM, BLOCK_DIM, 1);
    sycl::range<3> grid0(query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, 1);
    if (query_nb % BLOCK_DIM != 0)
        grid0[0] += 1;
    if (ref_nb   % BLOCK_DIM != 0)
        grid0[1] += 1;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        cl::sycl::stream out(1024, 256, cgh);

        sycl::range<2> shared_A_range_ct1(16 /*BLOCK_DIM*/, 16 /*BLOCK_DIM*/);
        sycl::range<2> shared_B_range_ct1(16 /*BLOCK_DIM*/, 16 /*BLOCK_DIM*/);

        sycl::accessor<float, 2, sycl::access::mode::read_write,
                       sycl::access::target::local>
            shared_A_acc_ct1(shared_A_range_ct1, cgh);
        sycl::accessor<float, 2, sycl::access::mode::read_write,
                       sycl::access::target::local>
            shared_B_acc_ct1(shared_B_range_ct1, cgh);
        sycl::accessor<int, 0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            begin_A_acc_ct1(cgh);
        sycl::accessor<int, 0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            begin_B_acc_ct1(cgh);
        sycl::accessor<int, 0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            step_A_acc_ct1(cgh);
        sycl::accessor<int, 0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            step_B_acc_ct1(cgh);
        sycl::accessor<int, 0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            end_A_acc_ct1(cgh);

        auto dpct_global_range = grid0 * block0;
        out << "ref_dev_sycl" << ref_dev << "\n";

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block0.get(2), block0.get(1), block0.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                compute_distances(
                    ref_dev, ref_nb, ref_pitch, query_dev, query_nb,
                    query_pitch, dim, dist_dev, item_ct1,
                    dpct::accessor<float, dpct::local, 2>(shared_A_acc_ct1,
                                                          shared_A_range_ct1),
                    dpct::accessor<float, dpct::local, 2>(shared_B_acc_ct1,
                                                          shared_B_range_ct1),
                    begin_A_acc_ct1.get_pointer(),
                    begin_B_acc_ct1.get_pointer(), step_A_acc_ct1.get_pointer(),
                    step_B_acc_ct1.get_pointer(), end_A_acc_ct1.get_pointer());
            });
    });
    /*
    DPCT1010:8: SYCL uses exceptions to report errors and does not use the error
    codes. The call was replaced with 0. You need to rewrite this code.
    */
    if (0 != 0) {
        printf("ERROR: Unable to execute kernel\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        return false;
    }

    // Sort the distances with their respective indexes
    sycl::range<3> block1(256, 1, 1);
    sycl::range<3> grid1(query_nb / 256, 1, 1);
    if (query_nb % 256 != 0)
        grid1[0] += 1;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid1 * block1;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block1.get(2), block1.get(1), block1.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                modified_insertion_sort(dist_dev, dist_pitch, index_dev,
                                        index_pitch, query_nb, ref_nb, k,
                                        item_ct1);
            });
    });
    /*
    DPCT1010:9: SYCL uses exceptions to report errors and does not use the error
    codes. The call was replaced with 0. You need to rewrite this code.
    */
    if (0 != 0) {
        printf("ERROR: Unable to execute kernel\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        return false;
    }

    // Compute the square root of the k smallest distances
    sycl::range<3> block2(16, 16, 1);
    sycl::range<3> grid2(query_nb / 16, k / 16, 1);
    if (query_nb % 16 != 0)
        grid2[0] += 1;
    if (k % 16 != 0)
        grid2[1] += 1;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid2 * block2;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block2.get(2), block2.get(1), block2.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                compute_sqrt(dist_dev, query_nb, query_pitch, k, item_ct1);
            });
    });
    /*
    DPCT1010:10: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    if (0 != 0) {
        printf("ERROR: Unable to execute kernel\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        return false;
    }

    // Copy k smallest distances / indexes from the device to the host
    /*
    DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 = (dpct::dpct_memcpy(knn_dist, query_nb * size_of_float, dist_dev,
                              dist_pitch_in_bytes, query_nb * size_of_float, k,
                              dpct::device_to_host),
            0);
    /*
    DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err1 = (dpct::dpct_memcpy(knn_index, query_nb * size_of_int, index_dev,
                              index_pitch_in_bytes, query_nb * size_of_int, k,
                              dpct::device_to_host),
            0);
    if (err0 != 0 || err1 != 0) {
        printf("ERROR: Unable to copy data from device to host\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        return false; 
    }

    // Memory clean-up
    sycl::free(ref_dev, dpct::get_default_context());
    sycl::free(query_dev, dpct::get_default_context());
    sycl::free(dist_dev, dpct::get_default_context());
    sycl::free(index_dev, dpct::get_default_context());

    return true;
}
catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
//    std::exit(1);
    return 1;
}

bool knn_cuda_texture(const float *ref, int ref_nb, const float *query,
                      int query_nb, int dim, int k, float *knn_dist,
                      int *knn_index) try {

    // Constants
    unsigned int size_of_float = sizeof(float);
    unsigned int size_of_int   = sizeof(int);   

    // Return variables
    int err0, err1, err2;

    // Check that we have at least one CUDA device 
    int nb_devices;
    /*
    DPCT1003:21: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 = (nb_devices = dpct::dev_mgr::instance().device_count(), 0);
    if (err0 != 0 || nb_devices == 0) {
        printf("ERROR: No CUDA device found\n");
        return false;
    }

    // Select the first CUDA device as default
    /*
    DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 = (dpct::dev_mgr::instance().select_device(0), 0);

    // Allocate global memory
    float * query_dev = NULL;
    float * dist_dev  = NULL;
    int *   index_dev = NULL;
    size_t  query_pitch_in_bytes;
    size_t  dist_pitch_in_bytes;
    size_t  index_pitch_in_bytes;
    /*
    DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 = (dpct::dpct_malloc((void **)&query_dev, &query_pitch_in_bytes,
                              query_nb * size_of_float, dim),
            0);
    /*
    DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err1 = (dpct::dpct_malloc((void **)&dist_dev, &dist_pitch_in_bytes,
                              query_nb * size_of_float, ref_nb),
            0);
    /*
    DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err2 = (dpct::dpct_malloc((void **)&index_dev, &index_pitch_in_bytes,
                              query_nb * size_of_int, k),
            0);
    if (err0 != 0 || err1 != 0 || err2 != 0) {
        printf("ERROR: Memory allocation error (cudaMallocPitch)\n");
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        return false;
    }

    // Deduce pitch values
    size_t query_pitch = query_pitch_in_bytes / size_of_float;
    size_t dist_pitch  = dist_pitch_in_bytes  / size_of_float;
    size_t index_pitch = index_pitch_in_bytes / size_of_int;

    // Check pitch values
    if (query_pitch != dist_pitch || query_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        return false; 
    }

    // Copy query data from the host to the device
    /*
    DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 =
        (dpct::dpct_memcpy(query_dev, query_pitch_in_bytes, query,
                           query_nb * size_of_float, query_nb * size_of_float,
                           dim, dpct::host_to_device),
         0);
    /*
    DPCT1000:14: Error handling if-stmt was detected but could not be rewritten.
    */
    if (err0 != 0) {
        printf("ERROR: Unable to copy data from host to device\n");
        /*
        DPCT1001:13: The statement could not be removed.
        */
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        return false; 
    }

    // Allocate CUDA array for reference points
    //modified::
    //cudaArray* ref_array_dev = NULL;
    dpct::image_matrix *ref_array_dev;
    dpct::image_channel channel_desc =
        dpct::create_image_channel(32, 0, 0, 0, dpct::channel_float);
    /*
    DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 = (ref_array_dev = new dpct::image_matrix(channel_desc,
                                                   sycl::range<2>(ref_nb, dim)),
            0);
    /*
    DPCT1000:16: Error handling if-stmt was detected but could not be rewritten.
    */
    if (err0 != 0) {
        printf("ERROR: Memory allocation error (cudaMallocArray)\n");
        /*
        DPCT1001:15: The statement could not be removed.
        */
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        return false; 
    }

    // Copy reference points from host to device
    /*
    DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 = (dpct::dpct_memcpy(
                ref_array_dev->to_pitched_data(), sycl::id<3>(0, 0, 0),
                dpct::pitched_data((void*)ref, ref_nb * size_of_float * dim,
                                   ref_nb * size_of_float * dim, 1),
                sycl::id<3>(0, 0, 0),
                sycl::range<3>(ref_nb * size_of_float * dim, 1, 1)),
            0);
    /*
    DPCT1000:18: Error handling if-stmt was detected but could not be rewritten.
    */
    if (err0 != 0) {
        printf("ERROR: Unable to copy data from host to device\n");
        /*
        DPCT1001:17: The statement could not be removed.
        */
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        delete ref_array_dev;
        return false; 
    }

    // Resource descriptor
    dpct::image_data res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.type = dpct::data_matrix;
    res_desc.data.matrix = ref_array_dev;

    // Texture descriptor
    dpct::image_info tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addr_mode() = sycl::addressing_mode::clamp_to_edge;
    tex_desc.addr_mode() = sycl::addressing_mode::clamp_to_edge;
    tex_desc.filter_mode() = sycl::filtering_mode::nearest;
    /*
    DPCT1004:29: Could not generate replacement.
    */
    //tex_desc.readMode = cudaReadModeElementType;
    /*
    DPCT1004:30: Could not generate replacement.
    */
    //tex_desc.normalizedCoords = 0;

    // Create the texture
    dpct::image_base_p ref_tex_dev = 0;
    err0 = (dpct::create_image(&ref_tex_dev, &res_desc, &tex_desc), 0);
    /*
    DPCT1000:20: Error handling if-stmt was detected but could not be rewritten.
    */
    if (err0 != 0) {
        printf("ERROR: Unable to create the texture\n");
        /*
        DPCT1001:19: The statement could not be removed.
        */
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        delete ref_array_dev;
        return false; 
    }

    // Compute the squared Euclidean distances
    sycl::range<3> block0(16, 16, 1);
    sycl::range<3> grid0(query_nb / 16, ref_nb / 16, 1);
    if (query_nb % 16 != 0)
        grid0[0] += 1;
    if (ref_nb   % 16 != 0)
        grid0[1] += 1;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto ref_tex_dev_acc =
            static_cast<dpct::image<float, 2> *>(ref_tex_dev)->get_access(cgh);

        auto dpct_global_range = grid0 * block0;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block0.get(2), block0.get(1), block0.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                compute_distance_texture(ref_tex_dev_acc, ref_nb, query_dev,
                                         query_nb, query_pitch, dim, dist_dev,
                                         item_ct1);
            });
    });
    /*
    DPCT1010:31: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    if (0 != 0) {
        printf("ERROR: Unable to execute kernel\n");
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        delete ref_array_dev;
        dpct::dpct_free(ref_tex_dev);
        return false;
    }

    // Sort the distances with their respective indexes
    sycl::range<3> block1(256, 1, 1);
    sycl::range<3> grid1(query_nb / 256, 1, 1);
    if (query_nb % 256 != 0)
        grid1[0] += 1;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid1 * block1;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block1.get(2), block1.get(1), block1.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                modified_insertion_sort(dist_dev, dist_pitch, index_dev,
                                        index_pitch, query_nb, ref_nb, k,
                                        item_ct1);
            });
    });
    /*
    DPCT1010:32: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    if (0 != 0) {
        printf("ERROR: Unable to execute kernel\n");
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        delete ref_array_dev;
        dpct::dpct_free(ref_tex_dev);
        return false;
    }

    // Compute the square root of the k smallest distances
    sycl::range<3> block2(16, 16, 1);
    sycl::range<3> grid2(query_nb / 16, k / 16, 1);
    if (query_nb % 16 != 0)
        grid2[0] += 1;
    if (k % 16 != 0)
        grid2[1] += 1;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid2 * block2;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block2.get(2), block2.get(1), block2.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                compute_sqrt(dist_dev, query_nb, query_pitch, k, item_ct1);
            });
    });
    /*
    DPCT1010:33: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    if (0 != 0) {
        printf("ERROR: Unable to execute kernel\n");
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        delete ref_array_dev;
        dpct::dpct_free(ref_tex_dev);
        return false;
    }

    // Copy k smallest distances / indexes from the device to the host
    /*
    DPCT1003:34: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 = (dpct::dpct_memcpy(knn_dist, query_nb * size_of_float, dist_dev,
                              dist_pitch_in_bytes, query_nb * size_of_float, k,
                              dpct::device_to_host),
            0);
    /*
    DPCT1003:35: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err1 = (dpct::dpct_memcpy(knn_index, query_nb * size_of_int, index_dev,
                              index_pitch_in_bytes, query_nb * size_of_int, k,
                              dpct::device_to_host),
            0);
    if (err0 != 0 || err1 != 0) {
        printf("ERROR: Unable to copy data from device to host\n");
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        delete ref_array_dev;
        dpct::dpct_free(ref_tex_dev);
        return false; 
    }

    // Memory clean-up
    sycl::free(query_dev, dpct::get_default_context());
    sycl::free(dist_dev, dpct::get_default_context());
    sycl::free(index_dev, dpct::get_default_context());
    delete ref_array_dev;
    dpct::dpct_free(ref_tex_dev);

    return true;
}
catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

bool knn_cublas(const float *ref, int ref_nb, const float *query, int query_nb,
                int dim, int k, float *knn_dist, int *knn_index) try {

    // Constants
    const unsigned int size_of_float = sizeof(float);
    const unsigned int size_of_int   = sizeof(int);

    // Return variables
    int err0, err1, err2, err3, err4, err5;

    // Check that we have at least one CUDA device 
    int nb_devices;
    /*
    DPCT1003:38: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 = (nb_devices = dpct::dev_mgr::instance().device_count(), 0);
    if (err0 != 0 || nb_devices == 0) {
        printf("ERROR: No CUDA device found\n");
        return false;
    }

    // Select the first CUDA device as default
    /*
    DPCT1003:39: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 = (dpct::dev_mgr::instance().select_device(0), 0);

    // Initialize CUBLAS
    /*
    DPCT1026:36: The call to cublasInit was removed, because this call is
    redundant in DPC++.
    */

    // Allocate global memory
    float * ref_dev        = NULL;
    float * query_dev      = NULL;
    float * dist_dev       = NULL;
    int   * index_dev      = NULL;
    float * ref_norm_dev   = NULL;
    float * query_norm_dev = NULL;
    size_t  ref_pitch_in_bytes;
    size_t  query_pitch_in_bytes;
    size_t  dist_pitch_in_bytes;
    size_t  index_pitch_in_bytes;
    /*
    DPCT1003:40: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 = (dpct::dpct_malloc((void **)&ref_dev, &ref_pitch_in_bytes,
                              ref_nb * size_of_float, dim),
            0);
    /*
    DPCT1003:41: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err1 = (dpct::dpct_malloc((void **)&query_dev, &query_pitch_in_bytes,
                              query_nb * size_of_float, dim),
            0);
    /*
    DPCT1003:42: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err2 = (dpct::dpct_malloc((void **)&dist_dev, &dist_pitch_in_bytes,
                              query_nb * size_of_float, ref_nb),
            0);
    /*
    DPCT1003:43: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err3 = (dpct::dpct_malloc((void **)&index_dev, &index_pitch_in_bytes,
                              query_nb * size_of_int, k),
            0);
    /*
    DPCT1003:44: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err4 = (ref_norm_dev = (float *)sycl::malloc_device(
                ref_nb * size_of_float, dpct::get_current_device(),
                dpct::get_default_context()),
            0);
    /*
    DPCT1003:45: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err5 = (query_norm_dev = (float *)sycl::malloc_device(
                query_nb * size_of_float, dpct::get_current_device(),
                dpct::get_default_context()),
            0);
    if (err0 != 0 || err1 != 0 || err2 != 0 || err3 != 0 || err4 != 0 ||
        err5 != 0) {
        printf("ERROR: Memory allocation error\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        sycl::free(ref_norm_dev, dpct::get_default_context());
        sycl::free(query_norm_dev, dpct::get_default_context());
        /*
        DPCT1026:46: The call to cublasShutdown was removed, because this call
        is redundant in DPC++.
        */
        return false;
    }

    // Deduce pitch values
    size_t ref_pitch   = ref_pitch_in_bytes   / size_of_float;
    size_t query_pitch = query_pitch_in_bytes / size_of_float;
    size_t dist_pitch  = dist_pitch_in_bytes  / size_of_float;
    size_t index_pitch = index_pitch_in_bytes / size_of_int;

    // Check pitch values
    if (query_pitch != dist_pitch || query_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        sycl::free(ref_norm_dev, dpct::get_default_context());
        sycl::free(query_norm_dev, dpct::get_default_context());
        /*
        DPCT1026:47: The call to cublasShutdown was removed, because this call
        is redundant in DPC++.
        */
        return false;
    }

    // Copy reference and query data from the host to the device
    /*
    DPCT1003:48: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 = (dpct::dpct_memcpy(ref_dev, ref_pitch_in_bytes, ref,
                              ref_nb * size_of_float, ref_nb * size_of_float,
                              dim, dpct::host_to_device),
            0);
    /*
    DPCT1003:49: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err1 =
        (dpct::dpct_memcpy(query_dev, query_pitch_in_bytes, query,
                           query_nb * size_of_float, query_nb * size_of_float,
                           dim, dpct::host_to_device),
         0);
    if (err0 != 0 || err1 != 0) {
        printf("ERROR: Unable to copy data from host to device\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        sycl::free(ref_norm_dev, dpct::get_default_context());
        sycl::free(query_norm_dev, dpct::get_default_context());
        /*
        DPCT1026:50: The call to cublasShutdown was removed, because this call
        is redundant in DPC++.
        */
        return false;
    }

    // Compute the squared norm of the reference points
    sycl::range<3> block0(256, 1, 1);
    sycl::range<3> grid0(ref_nb / 256, 1, 1);
    if (ref_nb % 256 != 0)
        grid0[0] += 1;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid0 * block0;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block0.get(2), block0.get(1), block0.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                compute_squared_norm(ref_dev, ref_nb, ref_pitch, dim,
                                     ref_norm_dev, item_ct1);
            });
    });
    /*
    DPCT1010:51: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    if (0 != 0) {
        printf("ERROR: Unable to execute kernel\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        sycl::free(ref_norm_dev, dpct::get_default_context());
        sycl::free(query_norm_dev, dpct::get_default_context());
        /*
        DPCT1026:52: The call to cublasShutdown was removed, because this call
        is redundant in DPC++.
        */
        return false;
    }

    // Compute the squared norm of the query points
    sycl::range<3> block1(256, 1, 1);
    sycl::range<3> grid1(query_nb / 256, 1, 1);
    if (query_nb % 256 != 0)
        grid1[0] += 1;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid1 * block1;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block1.get(2), block1.get(1), block1.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                compute_squared_norm(query_dev, query_nb, query_pitch, dim,
                                     query_norm_dev, item_ct1);
            });
    });
    /*
    DPCT1010:53: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    if (0 != 0) {
        printf("ERROR: Unable to execute kernel\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        sycl::free(ref_norm_dev, dpct::get_default_context());
        sycl::free(query_norm_dev, dpct::get_default_context());
        /*
        DPCT1026:54: The call to cublasShutdown was removed, because this call
        is redundant in DPC++.
        */
        return false;
    }

    // Computation of query*transpose(reference)
    mkl::blas::gemm(dpct::get_default_queue(), mkl::transpose::nontrans,
                    mkl::transpose::trans, (int)query_pitch, (int)ref_pitch,
                    dim, (float)-2.0, query_dev, query_pitch, ref_dev,
                    ref_pitch, (float)0.0, dist_dev, query_pitch)
        .wait();
    /*
    DPCT1027:55: The call to cublasGetError was replaced with 0, because this
    call is redundant in DPC++.
    */
    if (0 != 0) {
        printf("ERROR: Unable to execute cublasSgemm\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        sycl::free(ref_norm_dev, dpct::get_default_context());
        sycl::free(query_norm_dev, dpct::get_default_context());
        /*
        DPCT1026:56: The call to cublasShutdown was removed, because this call
        is redundant in DPC++.
        */
        return false;
    }

    // Add reference points norm
    sycl::range<3> block2(16, 16, 1);
    sycl::range<3> grid2(query_nb / 16, ref_nb / 16, 1);
    if (query_nb % 16 != 0)
        grid2[0] += 1;
    if (ref_nb   % 16 != 0)
        grid2[1] += 1;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            shared_vec_acc_ct1(sycl::range<1>(16), cgh);

        auto dpct_global_range = grid2 * block2;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block2.get(2), block2.get(1), block2.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                add_reference_points_norm(dist_dev, query_nb, dist_pitch,
                                          ref_nb, ref_norm_dev, item_ct1,
                                          shared_vec_acc_ct1.get_pointer());
            });
    });
    /*
    DPCT1010:57: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    if (0 != 0) {
        printf("ERROR: Unable to execute kernel\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        sycl::free(ref_norm_dev, dpct::get_default_context());
        sycl::free(query_norm_dev, dpct::get_default_context());
        /*
        DPCT1026:58: The call to cublasShutdown was removed, because this call
        is redundant in DPC++.
        */
        return false;
    }

    // Sort each column
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid1 * block1;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block1.get(2), block1.get(1), block1.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                modified_insertion_sort(dist_dev, dist_pitch, index_dev,
                                        index_pitch, query_nb, ref_nb, k,
                                        item_ct1);
            });
    });
    /*
    DPCT1010:59: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    if (0 != 0) {
        printf("ERROR: Unable to execute kernel\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        sycl::free(ref_norm_dev, dpct::get_default_context());
        sycl::free(query_norm_dev, dpct::get_default_context());
        /*
        DPCT1026:60: The call to cublasShutdown was removed, because this call
        is redundant in DPC++.
        */
        return false;
    }

    // Add query norm and compute the square root of the of the k first elements
    sycl::range<3> block3(16, 16, 1);
    sycl::range<3> grid3(query_nb / 16, k / 16, 1);
    if (query_nb % 16 != 0)
        grid3[0] += 1;
    if (k        % 16 != 0)
        grid3[1] += 1;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid3 * block3;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block3.get(2), block3.get(1), block3.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                add_query_points_norm_and_sqrt(dist_dev, query_nb, dist_pitch,
                                               k, query_norm_dev, item_ct1);
            });
    });
    /*
    DPCT1010:61: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    if (0 != 0) {
        printf("ERROR: Unable to execute kernel\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        sycl::free(ref_norm_dev, dpct::get_default_context());
        sycl::free(query_norm_dev, dpct::get_default_context());
        /*
        DPCT1026:62: The call to cublasShutdown was removed, because this call
        is redundant in DPC++.
        */
        return false;
    }

    // Copy k smallest distances / indexes from the device to the host
    /*
    DPCT1003:63: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err0 = (dpct::dpct_memcpy(knn_dist, query_nb * size_of_float, dist_dev,
                              dist_pitch_in_bytes, query_nb * size_of_float, k,
                              dpct::device_to_host),
            0);
    /*
    DPCT1003:64: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    err1 = (dpct::dpct_memcpy(knn_index, query_nb * size_of_int, index_dev,
                              index_pitch_in_bytes, query_nb * size_of_int, k,
                              dpct::device_to_host),
            0);
    if (err0 != 0 || err1 != 0) {
        printf("ERROR: Unable to copy data from device to host\n");
        sycl::free(ref_dev, dpct::get_default_context());
        sycl::free(query_dev, dpct::get_default_context());
        sycl::free(dist_dev, dpct::get_default_context());
        sycl::free(index_dev, dpct::get_default_context());
        sycl::free(ref_norm_dev, dpct::get_default_context());
        sycl::free(query_norm_dev, dpct::get_default_context());
        /*
        DPCT1026:65: The call to cublasShutdown was removed, because this call
        is redundant in DPC++.
        */
        return false;
    }

    // Memory clean-up and CUBLAS shutdown
    sycl::free(ref_dev, dpct::get_default_context());
    sycl::free(query_dev, dpct::get_default_context());
    sycl::free(dist_dev, dpct::get_default_context());
    sycl::free(index_dev, dpct::get_default_context());
    sycl::free(ref_norm_dev, dpct::get_default_context());
    sycl::free(query_norm_dev, dpct::get_default_context());
    /*
    DPCT1026:37: The call to cublasShutdown was removed, because this call is
    redundant in DPC++.
    */

    return true;
}
catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}
