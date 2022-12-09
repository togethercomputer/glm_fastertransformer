[ ! -f /FasterTransformer/build/gemm_config.in ] && cd /FasterTransformer/build && ./bin/gpt_gemm 1 1 128 96 128 49152 150528 1 8
cd /app && bash glm/glm_server.sh