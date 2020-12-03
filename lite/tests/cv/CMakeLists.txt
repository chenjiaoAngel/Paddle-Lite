<<<<<<< HEAD
if((NOT LITE_WITH_OPENCL AND NOT LITE_WITH_FPGA) AND LITE_WITH_ARM)
    lite_cc_test(image_convert_test SRCS image_convert_test.cc DEPS cv_arm ${lite_cv_deps} ${arm_kernels} ${lite_ops} ${host_kernels})
=======
if(LITE_WITH_CV AND (NOT LITE_WITH_OPENCL AND NOT LITE_WITH_FPGA AND NOT LITE_WITH_MLU) AND LITE_WITH_ARM)
    lite_cc_test(image_convert_test SRCS image_convert_test.cc DEPS paddle_cv_arm)
    lite_cc_test(image_profiler_test SRCS image_profiler_test.cc DEPS paddle_cv_arm anakin_cv_arm)
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
endif()
