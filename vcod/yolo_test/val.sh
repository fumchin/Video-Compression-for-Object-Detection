#! /bin/bash
# python validate_q.py --yaml VOC_val.yaml > origin_result.txt
# python validate_q.py --yaml VOC_q1.yaml > q1_result.txt
# python validate_q.py --yaml VOC_q2.yaml > q2_result.txt
python validate.py --yaml VOC_q3.yaml > q3_stage2_best.txt
# python validate_q6_stage2.py --yaml VOC_q6_stage2.yaml > q6_test_stage2_e-6_result_dual_ep0.txt
# python validate_q6_stage2.py --yaml VOC_q6_stage2.yaml > q6_test_2.txt
# python validate_q.py --yaml VOC_q4.yaml > q4_result.txt
# python validate_q.py --yaml VOC_q5.yaml > q5_result.txt