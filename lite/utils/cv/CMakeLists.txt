<<<<<<< HEAD
if((NOT LITE_WITH_OPENCL AND NOT LITE_WITH_FPGA) AND LITE_WITH_ARM)
    set(lite_cv_deps)
    lite_cc_library(cv_arm SRCS
            image_convert.cc
            image_preprocess.cc
            image2tensor.cc
            image_flip.cc
            image_rotate.cc
            image_transform.cc
            DEPS ${lite_cv_deps} tensor)
=======
if(LITE_WITH_CV AND LITE_WITH_ARM)
    lite_cc_library(paddle_cv_arm SRCS
            image_convert.cc
            bgr_rotate.cc
            paddle_image_preprocess.cc
            image2tensor.cc
            image_flip.cc
            image_rotate.cc
            image_resize.cc
            DEPS paddle_api place)
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
endif()
