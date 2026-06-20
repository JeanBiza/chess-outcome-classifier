import torch
from model import ChessOutcomeCNN

model = ChessOutcomeCNN()
model.load_state_dict(torch.load("best_chess_model.pt", map_location="cpu"))
model.eval()

dummy_board = torch.zeros(1, 12, 8, 8)
dummy_extra = torch.zeros(1, 7)

torch.onnx.export(
    model, (dummy_board, dummy_extra),
    "chess_model.onnx",
    input_names=["board", "extra"],
    output_names=["logits"],
    dynamic_axes={"board": {0: "batch"}, "extra": {0: "batch"}, "logits": {0: "batch"}}
)
print("Exportado a chess_model.onnx")