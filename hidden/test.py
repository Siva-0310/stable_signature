import torch
from models import HiddenEncoder, HiddenDecoder

def run_test(is_residual, is_attn, r, num_bits=32, img_size=(3, 128, 128)):
    """
    Function to test HiddenEncoder and HiddenDecoder with given configurations.
    
    Args:
    - is_residual (bool): Use residual blocks or not.
    - is_attn (bool): Use CBAM attention or not.
    - r (int): Reduction ratio for CBAM.
    - num_bits (int): Number of bits in the watermark message.
    - img_size (tuple): Size of the input image (channels, height, width).
    """
    # Configurations for number of blocks
    encoder_blocks = 4 if not is_residual else 2
    decoder_blocks = 8 if not is_residual else 4
    channels = 64  # Number of channels in hidden layers
    
    # Create dummy data
    imgs = torch.rand((2, *img_size))  # Batch of 2 images
    msgs = torch.rand((2, num_bits))   # Batch of 2 watermark messages
    
    # Initialize encoder and decoder with the provided configuration
    encoder = HiddenEncoder(num_blocks=encoder_blocks, num_bits=num_bits, channels=channels,
                            r=r, is_attn=is_attn, is_residual=is_residual)
    
    decoder = HiddenDecoder(num_blocks=decoder_blocks, num_bits=num_bits, channels=channels,
                            r=r, is_attn=is_attn, is_residual=is_residual)
    
    # Print the encoder and decoder architectures
    print(f"Encoder Architecture: is_residual={is_residual}, is_attn={is_attn}, r={r}")
    print(encoder)  # Print encoder architecture
    
    print(f"Decoder Architecture: is_residual={is_residual}, is_attn={is_attn}, r={r}")
    print(decoder)  # Print decoder architecture
    
    # Run the encoder
    encoded_imgs = encoder(imgs, msgs)
    print(f"Encoder output shape: {encoded_imgs.shape}")
    
    # Run the decoder
    decoded_msgs = decoder(encoded_imgs)
    print(f"Decoder output shape: {decoded_msgs.shape}")
    
    # Verify shapes
    assert encoded_imgs.shape == (2, 3, img_size[1], img_size[2]), "Encoded image shape mismatch"
    assert decoded_msgs.shape == (2, num_bits), "Decoded message shape mismatch"
    
    print("Test passed!\n")


if __name__ == "__main__":
    # Test all combinations of is_residual and is_attn
    for is_residual in [False, True]:
        for is_attn in [False, True]:
            run_test(is_residual, is_attn, r=16)