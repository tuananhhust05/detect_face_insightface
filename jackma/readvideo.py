import PyNvCodec as nvc
import numpy as np

def read_video_with_pynvcodec(video_path):
    # Create a decoder object capable of reading the input video
    try:
        gpu_id = 0  # Select GPU device ID
        nv_decoder = nvc.PyNvDecoder(video_path, gpu_id)
        
        # Print video properties
        print('Video Bitrate:', nv_decoder.Bitrate())
        print('Video FPS:', nv_decoder.Framerate())

        while True:
            # Decode a single surface from the video
            raw_surface = nv_decoder.DecodeSingleSurface()
            if raw_surface.Empty():  # Check if surface is empty (indicating end of stream)
                break

            # Convert the NV12 surface to RGB
            to_rgb_converter = nvc.PySurfaceConverter(
                raw_surface.Width(), raw_surface.Height(),
                nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpu_id
            )
            rgb_surface = to_rgb_converter.Execute(raw_surface)

            # Copy the RGB data from GPU to CPU (numpy array)
            rgb_frame = np.ndarray(shape=(rgb_surface.Height() * rgb_surface.Pitch(), 3),
                                   dtype=np.uint8,
                                   buffer=rgb_surface.PlanePtr())
            
            # Process or display the frame as needed
            print('Decoded Frame:', rgb_frame.shape)
    
    except Exception as e:
        print(f'Error during decoding: {e}')

# Example usage
video_path = 'video.mp4'
read_video_with_pynvcodec(video_path)