import torch
from base_lcm import BaseLCM, SonarEncoder

def infer_model_config_from_ckpt(ckpt):
    prenet_w = ckpt["prenet.linear.weight"]
    postnet_w = ckpt["postnet.linear.weight"]
    ff_w = ckpt["transformer_decoder.layers.0.linear1.weight"]
    layer_ids = {
        int(k.split(".")[2])
        for k in ckpt.keys()
        if k.startswith("transformer_decoder.layers.")
    }

    input_dim = prenet_w.shape[1]
    hidden_dim = prenet_w.shape[0]
    output_dim = postnet_w.shape[0]
    ff_dim = ff_w.shape[0]
    num_layers = len(layer_ids)

    # Number of heads is not encoded explicitly in state_dict.
    # Use 8 when valid, otherwise pick the largest common fallback.
    if hidden_dim % 8 == 0:
        num_heads = 8
    elif hidden_dim % 4 == 0:
        num_heads = 4
    elif hidden_dim % 2 == 0:
        num_heads = 2
    else:
        num_heads = 1

    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "ff_dim": ff_dim,
        "output_dim": output_dim,
    }

def load_model(device, model_path="saved_models/base_lcm_model.pth"):
    ckpt = torch.load(model_path, map_location=device)
    cfg = infer_model_config_from_ckpt(ckpt)

    model = BaseLCM(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        output_dim=cfg["output_dim"],
    ).to(device)
    model.load_state_dict(ckpt)
    model.eval()

    print("Loaded checkpoint config:", cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of model parameters: {total_params}")
    return model

def infer_example(model, encoder, question, lang="en", device="cpu"):
    # Encode the question
    with torch.no_grad():
        embedding = encoder.encode([question], lang=lang, batch_size=1).to(device)
        # Model expects (batch, input_dim), possibly (batch, seq, input_dim). Shape accordingly.
        output = model(embedding)
        return output.cpu().numpy()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/saj/git/large_concept_model_tt/saved_models/base_lcm_model.pth"
    model = load_model(device, model_path)

    # Example question
    example_question = "What are the basic causes of climate change?"
    print("Question:", example_question)
    
    # Encoder must match the one used in training
    encoder = SonarEncoder(device=device)
    decoder = SonarEncoder(device=device)
    # Infer embedding for question
    result_embedding = infer_example(model, encoder, example_question, lang="en", device=device)
    print("Returned embedding shape:", result_embedding.shape)
    
    print("First 10 elements of the embedding:", result_embedding[0])


    
    # For demonstration, we could (if we have a set of candidate answers with embeddings) compare the result.
    # Here, we'll just show that we've produced a valid embedding for an input question.
