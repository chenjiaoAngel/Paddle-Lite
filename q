[1mdiff --git a/lite/core/mir/fusion/conv_conv_fuser.cc b/lite/core/mir/fusion/conv_conv_fuser.cc[m
[1mindex 2393ff5..ddd7cb8 100644[m
[1m--- a/lite/core/mir/fusion/conv_conv_fuser.cc[m
[1m+++ b/lite/core/mir/fusion/conv_conv_fuser.cc[m
[36m@@ -132,6 +132,15 @@[m [mvoid ConvConvFuser::BuildPattern() {[m
               VLOG(5) << "The kernel size of the second conv must be 1x1";[m
               continue;[m
             }[m
[32m+[m[32m            bool flag_pad = false;[m
[32m+[m[32m            for (int i = 0; i < paddings1.size(); i++) {[m
[32m+[m[32m              if (paddings1[i] != 0) {[m
[32m+[m[32m                LOG(INFO) << "paddings1 is not eauql 0, but is " << paddings1[i];[m
[32m+[m[32m                flag_pad = true;[m
[32m+[m[32m                break;[m
[32m+[m[32m              }[m
[32m+[m[32m            }[m
[32m+[m[32m            if (flag_pad) continue;[m
             if (groups0 != 1 || groups1 != 1) {[m
               VLOG(5) << "The all groups of weight_dim must be 1";[m
               continue;[m
[36m@@ -147,14 +156,16 @@[m [mvoid ConvConvFuser::BuildPattern() {[m
             // computation: ic0 x (oc1-oc0) < oc0 x oc1[m
             VLOG(5) << "a: " << (ch_in_0 * (ch_out_1 - ch_out_0)) << " <= "[m
                     << "b: " << (ch_out_0 * ch_out_1);[m
[31m-[m
[31m-            if (ch_in_0 * (ch_out_1 - ch_out_0) > ch_out_0 * ch_out_1) {[m
[32m+[m[32m            auto mula = (ch_in_0 * (ch_out_1 - ch_out_0));[m
[32m+[m[32m            auto mulb = (ch_out_0 * ch_out_1);[m
[32m+[m[32m            if (mula <= 0 || mula  > mulb) {[m
               VLOG(5) << "it dose not meet the requirment of conv+conv fusion "[m
                       << "computation "[m
[31m-                      << "a: " << (ch_in_0 * (ch_out_1 - ch_out_0)) << " <= "[m
[31m-                      << "b: " << (ch_out_0 * ch_out_1);[m
[32m+[m[32m                      << "a: " << mula << " <= "[m
[32m+[m[32m                      << "b: " << mulb;[m
               continue;[m
             }[m
[32m+[m[32m            VLOG(5) << "a: " << mula << ", b: " <<mulb;[m
             // create pattern[m
             VLOG(5) << "matched: " << conv_type0_ << " and " << conv_type1_;[m
             createPattern();[m
