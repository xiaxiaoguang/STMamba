import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def test_mamba_lm_head_model():
    # Define the configuration for the model
    config = MambaConfig(
        d_model=64,
        n_layer=4,
        d_intermediate=128,
        vocab_size=1000,  # This will be adjusted inside the model
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        pad_vocab_size_multiple=1,
        feature_size=32,  # New field for the feature size
        tie_embeddings=False,  # Not used, but should be set
    )

    # Create the model
    model = MambaLMHeadModel(config)
    model.cuda()

    # Define the input tensor with shape [batch_size, input_length, feature_size]
    batch_size = 2
    input_length = 10
    feature_size = 32
    input_tensor = torch.randn(batch_size, input_length, feature_size).cuda()

    # Define the output length
    output_length = 5

    # Run the model
    model.eval()
    with torch.no_grad():
        output_tensor = model.generate(input_tensor, output_length=output_length)
        output_tensor2 = model(input_tensor)

    # Check the shape of the output tensor
    assert output_tensor.shape == (batch_size, output_length, feature_size), \
        f"Expected output shape {(batch_size, output_length, feature_size)}, but got {output_tensor.shape}"
    
    assert output_tensor2.shape == (batch_size, input_length, feature_size), \
        f"Expected output shape {(batch_size, input_length, feature_size)}, but got {output_tensor.shape}"

    print("Test passed! Output tensor shape is correct.")

if __name__ == "__main__":
    test_mamba_lm_head_model()
