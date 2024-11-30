ffmpeg -i /home/ubuntu/liwen/ultimatevocalremover_api/assert/test_data/emma_original.mp4 \
    -map a /home/ubuntu/liwen/ultimatevocalremover_api/assert/result/emma_original.wav

ffmpeg -i /home/ubuntu/liwen/ultimatevocalremover_api/assert/test_data/chinese.mp4 \
    -map a /home/ubuntu/liwen/ultimatevocalremover_api/assert/result/chinese.wav
