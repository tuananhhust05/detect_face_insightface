import ffmpeg 

print("start")
 
ffmpeg.input("2.mp4", hwaccel='cuda', stream_loop=1).filter('fps', fps=60 ).output(f'imgs/frame_%04d.png', format='image2').run(overwrite_output=True)
print("finish")
